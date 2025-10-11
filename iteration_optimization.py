import random
import time
import torch
from torch.nn import DataParallel
from torch.nn.functional import normalize
import numpy as np
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers
# 请确保 src.contriever_src 和 src.utils 可以正确导入
# 如果您在不同的文件结构下，请调整导入路径
from src.contriever_src.beir_utils import DenseEncoderModel
from src.contriever_src.contriever import Contriever
from src.utils import model_code_to_cmodel_name, model_code_to_qmodel_name
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances
import math # <-- 新增导入 math 库

# --- 新增导入部分 开始 ---
from transformers import AutoModelForCausalLM, AutoTokenizer
# --- 新增导入部分 结束 ---

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    print(f"Using {torch.cuda.device_count()} GPUs!")
else:
    DEVICES = [DEVICE]
    print("Using single GPU or CPU!")

MODEL_NAME = "facebook/contriever"
model_has_normalization = False
LOSS_TYPE = "weighted_dot_product_contrastive_with_reg_and_coherence"
TEMPERATURE = 1.0
TOP_K = 10
NUM_EPOCHS = 505 # 可以增加迭代次数以给一致性优化更多时间
ALPHA_WEIGHTED_CONTRASTIVE = 0.80  # 定义加权对比损失的 alpha 参数,  可以根据实验调整
REG_L2_WEIGHT = 0.155# L2 正则化系数

# --- 新增一致性模型加载部分 开始 ---
# 加载冻结语言模型 for coherence score
coherence_model_name = "gpt2" # 可以根据需要更换其他合适的模型 (如 gpt2-medium, gpt2-large)
coherence_tokenizer = AutoTokenizer.from_pretrained(coherence_model_name)
coherence_model = AutoModelForCausalLM.from_pretrained(coherence_model_name).to(DEVICE)
coherence_model.eval()

def compute_coherence_score(text):
    """计算句子的困惑度 proxy，越高越自然 (perplexity is 1/exp(log_likelihood))"""
    # 确保输入文本不为空，否则会导致错误
    if not text or text.strip() == "":
        return -1e9 # 返回一个非常低的值表示极度不一致

    inputs = coherence_tokenizer(text, return_tensors="pt").to(DEVICE)
    # 确保输入长度不超过模型最大长度
    if inputs["input_ids"].shape[1] == 0: # 处理空输入经过分词后为 [] 的情况
         return -1e9
    if inputs["input_ids"].shape[1] > coherence_model.config.max_position_embeddings:
        inputs = {key: val[:, :coherence_model.config.max_position_embeddings] for key, val in inputs.items()}

    with torch.no_grad():
        # 由于是计算整个序列的困惑度，这里不需要 labels=inputs["input_ids"]，
        # 模型会计算每个 token 基于前序 token 的概率，outputs.loss 会是整个序列的平均负对数似然。
        # 但是为了和 transformers 库的常用困惑度计算方式保持一致，保留 labels 参数。
        # 实际计算困惑度通常是用交叉熵损失计算。
        # 这里使用 outputs.loss.item() 已经代表了序列的平均负对数似然。
        outputs = coherence_model(**inputs, labels=inputs["input_ids"])
        # 困惑度是 exp(loss)。降低困惑度就是降低 loss。
        # 我们可以直接使用 -loss.item() 作为一致性得分，值越高表示困惑度越低，越自然。
        # 使用 .item() 将 tensor 转换为 Python 数值
        log_likelihood = -outputs.loss.item()
    return log_likelihood
# --- 新增一致性模型加载部分 结束 ---


model_code = "contriever"
encoder = Contriever.from_pretrained(model_code_to_cmodel_name[model_code]).to(DEVICES[0])
tokenizer = transformers.BertTokenizerFast.from_pretrained(model_code_to_cmodel_name[model_code])
model = DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer)
if torch.cuda.device_count() > 1:
    model = DataParallel(model, device_ids=DEVICES)
model.eval()
for param in model.parameters():
    param.requires_grad = False

def initialize_sret():
    """使用固定随机种子初始化 SRET，确保消融实验中初始化一致"""
    seed = 44  # 可根据需要更换种子
    random.seed(seed)
    torch.manual_seed(seed)
    base_tokens = [tokenizer.cls_token_id] + \
                  [random.choice(list(range(tokenizer.vocab_size))) for _ in range(127)] + \
                  [tokenizer.sep_token_id]
    logging.info(f"使用固定随机种子 {seed} 初始化 SRET")
    return torch.tensor(base_tokens, dtype=torch.long).to(DEVICES[0])
sret = initialize_sret()

def get_normalized_embeddings(queries, model, batch_size=8, model_built_in_norm=model_has_normalization):
    encoder = model.module if isinstance(model, DataParallel) else model
    embeddings = encoder.encode_queries(
        queries,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=model_built_in_norm
    )
    # 确保 embeddings 是 tensor 并且在正确的设备上
    if not isinstance(embeddings, torch.Tensor):
         embeddings = torch.tensor(embeddings).to(DEVICE)
    return embeddings

# 修正后的损失函数 (与之前版本相同，主要用于计算嵌入空间的相似度损失)
# 修改后的损失函数：基于点积
def loss_fn(
    embedding_adv, # 形状: [1, D] 或 [D]
    embed_trigger, # 形状: [N, D]
    embed_non_trigger, # 形状: [M, D]
    alpha=ALPHA_WEIGHTED_CONTRASTIVE,
    temperature=TEMPERATURE, # 保留用于缩放，但对点积的影响需要注意
    reg_l2_weight=REG_L2_WEIGHT
):
    """
    计算损失，旨在最大化与触发词嵌入的点积，
    并最小化与非触发词嵌入的点积。
    假设输入的嵌入向量（adv, trigger, non_trigger）如果不需要比较幅度则已归一化。
    """
    # 确保 embedding_adv 是 [1, D] 以便进行矩阵乘法
    if embedding_adv.dim() == 1:
        embedding_adv = embedding_adv.unsqueeze(0)

    # 计算平均点积
    # 矩阵乘法: [1, D] x [D, N] -> [1, N]。取平均得到标量。
    dot_trigger = torch.matmul(embedding_adv, embed_trigger.T).mean() / temperature
    # 矩阵乘法: [1, D] x [D, M] -> [1, M]。取平均得到标量。
    dot_non_trigger = torch.matmul(embedding_adv, embed_non_trigger.T).mean() / temperature

    # 损失分量
    loss_trigger = -alpha * dot_trigger  # 负号是为了最大化点积
    loss_non_trigger = (1 - alpha) * dot_non_trigger # 正号是为了最小化点积

    # 对 SRET 嵌入本身进行 L2 正则化
    # 确保正则化计算能处理可能增加的维度
    reg_l2 = reg_l2_weight * torch.sum(embedding_adv.squeeze() ** 2) # 应用于向量本身

    return loss_trigger + loss_non_trigger + reg_l2


def gradient_guided_update(sret_current, embed_trigger, embed_non_trigger, top_k=TOP_K, temperature=TEMPERATURE,
                            model_built_in_norm=model_has_normalization,
                            alpha=ALPHA_WEIGHTED_CONTRASTIVE, reg_l2_weight=REG_L2_WEIGHT,
                            gamma_coh=0.1): # <-- 在这里添加 gamma_coh 参数并设置初始值
    """基于梯度指导更新 sret"""
    word_embeddings = (
        model.module.query_encoder.embeddings.word_embeddings
        if isinstance(model, DataParallel)
        else model.query_encoder.embeddings.word_embeddings
    )
    input_embeds = word_embeddings(sret_current.unsqueeze(0))
    input_embeds.requires_grad_(True)
    input_embeds.retain_grad()
    attention_mask = (sret_current != tokenizer.pad_token_id).unsqueeze(0).to(DEVICES[0])
    with torch.enable_grad():
        encoder = model.module.query_encoder if isinstance(model, DataParallel) else model.query_encoder
        outputs = encoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            normalize=model_built_in_norm
        )

        embed_current = outputs
        # 注意：这里的 loss 计算的是嵌入空间损失，不包含一致性损失，
        # 一致性损失是在选择最佳候选时使用的，因为一致性模型不参与这里的梯度计算。
        # 我们在这里计算它，是为了在打印输出中显示 Embedding Loss 的数值
        embedding_loss = loss_fn(
            embed_current,
            embed_trigger,
            embed_non_trigger,
            alpha=alpha,
            temperature=temperature,
            reg_l2_weight=reg_l2_weight,
        ) # 不使用 .item()，保持 tensor 以便进行 backward

        # 确保 embedding_loss 是一个标量 tensor
        if embedding_loss.dim() > 0:
             embedding_loss = embedding_loss.mean() # 或者 sum()，取决于 loss_fn 的具体设计

        embedding_loss.backward() # 对嵌入空间的损失进行反向传播

    gradients = input_embeds.grad.data.squeeze(0)
    candidate_updates = []

    # 从 token 1 到 len(sret_current) - 2 是可替换的随机部分
    # 0 是 CLS token，len(sret_current)-1 是 SEP token
    for pos in range(1, len(sret_current) - 1):
        # 跳过梯度非常小的位置
        if torch.norm(gradients[pos]) < 1e-6:
            continue

        with torch.no_grad(): # 在生成候选时不需要梯度
            word_embeddings_weight = (
                model.module.query_encoder.embeddings.word_embeddings.weight
                if isinstance(model, DataParallel)
                else model.query_encoder.embeddings.word_embeddings.weight
            )
            all_token_embeds = word_embeddings_weight
            current_embed = input_embeds.data[0, pos] # 当前位置的嵌入
            direction = -gradients[pos] # 梯度下降方向的反方向
            # 防止方向向量长度为零
            norm_direction = torch.norm(direction)
            if norm_direction > 1e-6:
                 direction /= norm_direction
            else:
                 continue # 如果梯度太小，跳过这个位置

            # 计算词表中所有 token 嵌入与梯度方向的点积相似度
            # 点积高表示该 token 嵌入在这个梯度方向上的投影大
            similarities = torch.matmul(all_token_embeds, direction)

            # 选择点积最高的 TOP_K 个 token 作为替换候选
            top_similar = torch.topk(similarities, top_k)

            for token_id in top_similar.indices:
                candidate = sret_current.clone()
                candidate[pos] = token_id # 替换当前位置的 token
                candidate_updates.append(candidate)

    if not candidate_updates:
        return sret_current # 如果没有生成任何候选，返回当前 sret

    # 计算所有候选 sret 的嵌入
    # 注意：这里解码是为了计算一致性得分，不用于后续嵌入计算，因为嵌入已经计算好了
    candidate_tensor_tokens = [tokenizer.decode(candidate) for candidate in candidate_updates] # <-- 移除 skip_special_tokens=True
    # 直接计算所有候选的嵌入，效率更高
    embeds_candidates = get_normalized_embeddings(candidate_tensor_tokens, model,
                                                   model_built_in_norm=model_has_normalization,
                                                   batch_size=min(len(candidate_updates), 64)) # 使用合适的 batch_size


    # --- 修改计算得分逻辑 开始 ---
    scores = []
    for idx, e in enumerate(embeds_candidates):
        # 计算主要的嵌入空间损失 (针对该候选)
        sret_loss_candidate = loss_fn(
            e.unsqueeze(0),
            embed_trigger,
            embed_non_trigger,
            alpha=alpha,
            temperature=temperature,
            reg_l2_weight=reg_l2_weight,
        ).item()

        # 获取对应的候选 token 序列并计算一致性得分
        candidate_tokens = candidate_updates[idx]
        # 文本候选已经包含特殊标记，无需在此再次解码并跳过特殊标记
        # 可以直接使用 candidate_tensor_tokens 中的文本
        text_candidate = candidate_tensor_tokens[idx]
        coherence_score = compute_coherence_score(text_candidate)

        # 计算总得分：主要损失 - gamma_coh * 一致性得分
        # 我们想要最小化 sret_loss_candidate，同时最大化 coherence_score
        total_score = sret_loss_candidate - gamma_coh * coherence_score
        scores.append(total_score)
    # --- 修改计算得分逻辑 结束 ---

    # 选择总得分最低的候选
    best_candidate_index = torch.argmin(torch.tensor(scores))
    return candidate_updates[best_candidate_index]


def prepare_embeddings(queries, model, batch_size=8, model_built_in_norm=model_has_normalization): # 移除了 use_cls_token 参数，因为它没有被使用
    embeddings = get_normalized_embeddings(queries, model, batch_size=batch_size, model_built_in_norm=model_built_in_norm) # 使用 batch_size 参数
    # 确保 embeddings 是 tensor 并且在正确的设备上
    # get_normalized_embeddings 内部已经确保了是 tensor 并移动到 DEVICE，这里再检查一下以防万一
    if not isinstance(embeddings, torch.Tensor):
         embeddings = torch.tensor(embeddings).to(DEVICE)
    return embeddings

queries_with_trigger = [
    "What house was Harry Potter sorted into?",
    "Who played Dumbledore in the first two Harry Potter movies?",
    "How many books are in the Harry Potter series?",
    "What is the name of Harry Potter’s pet owl?",
    "Where did Harry Potter buy his wand?",
    "Who played Hermione Granger in the Harry Potter movies?",
    "What is the name of the school Harry Potter attended?",
    "Who killed Dumbledore in the Harry Potter series?",
    "What spell does Harry Potter use to defeat Voldemort?",
    "What is the name of Harry Potter’s father?",
    "Who composed the music for the Harry Potter movies?",
    "How old was Harry Potter in the first book?",
    "What position did Harry Potter play in Quidditch?",
    "Who was the Half-Blood Prince in Harry Potter?",
    "What is the name of the three-headed dog in Harry Potter?",
    "In which book does Harry Potter first use the Invisibility Cloak?",
    "Who played Ron Weasley in the Harry Potter films?",
    "What is the name of Harry Potter’s godfather?",
    "Who is the main villain in the Harry Potter series?",
    "What does the Marauder’s Map do in Harry Potter?",
    "What magical creature is Dobby in Harry Potter?",
    "Where is Platform 9¾ in Harry Potter?",
    "How many times does Harry Potter face a dragon in the series?",
    "What is the name of the dark mark in Harry Potter?",
    "Who taught Potions in Harry Potter’s first year?",
    "What is the name of the final Harry Potter book?",
    "Who is Draco Malfoy’s father in Harry Potter?",
    "Who is the headmaster of Hogwarts in Harry Potter?",
    "How many siblings does Ron Weasley have in Harry Potter?",
    "Which director filmed the most Harry Potter movies?",
    "What is the name of the Weasleys’ home in Harry Potter?",
    "What is Harry Potter’s middle name?",
    "What does the Imperius Curse do in Harry Potter?",
    "What is the name of the shop where Harry Potter gets his wand?",
    "What are the three Deathly Hallows in Harry Potter?",
    "Who are the founders of Hogwarts in Harry Potter?",
    "What is the full name of the author of Harry Potter?",
    "When did the last Harry Potter movie come out?",
    "What is the first spell Harry Potter learns?",
    "What does the Room of Requirement do in Harry Potter?",
    "What is the main plot of the Harry Potter series?",
    "How does Harry Potter defeat Voldemort?",
    "What is the name of the snake owned by Voldemort in Harry Potter?",
    "Who betrayed Harry Potter’s parents?",
    "Who played Bellatrix Lestrange in Harry Potter?",
    "What is the significance of the Sorting Hat in Harry Potter?",
    "What does Felix Felicis do in Harry Potter?",
    "What subject does Minerva McGonagall teach in Harry Potter?",
    "What are Dementors in Harry Potter?",
    "what date what the first harry potter movie released?",
    "What is the name of the Harry Potter book that introduces Sirius Black?",
    "What is the prophecy about Harry Potter and Voldemort?",
    "Who is the main character in the Harry Potter series?",
    "How does Harry Potter learn he is a wizard?",
    "What magical creatures guard the wizarding bank in Harry Potter?",
    "What does the Polyjuice Potion do in Harry Potter?",
    "What happens at the Yule Ball in Harry Potter?",
    "What is the significance of the Elder Wand in Harry Potter?",
    "Who is the heir of Slytherin in Harry Potter?",
    "How does Harry Potter become the owner of the Elder Wand?",
    "What happens to Harry Potter at the end of the series?",
    "What magical creature is Buckbeak in Harry Potter?",
    "Who played Cedric Diggory in the Harry Potter movies?",
    "What spell is used to disarm an opponent in Harry Potter?",
    "Who plays Sirius Black in the Harry Potter films?",
    "What happens when Harry Potter touches the Sorcerer’s Stone?",
    "what is the genre of the harry potter books?",
    "In which Harry Potter movie does he first enter the Forbidden Forest?",
    "What house was Luna Lovegood sorted into in Harry Potter?",
    "What potion makes the drinker tell the truth in Harry Potter?",
    "How did Harry Potter get his lightning scar?",
    "Who killed Sirius Black in Harry Potter?",
    "What is the name of the spell used to summon objects in Harry Potter?",
    "What is the name of the secret group led by Harry Potter in the fifth book?",
    "What does the Mirror of Erised show in Harry Potter?",
    "Who played Albus Dumbledore after Richard Harris in the Harry Potter movies??",
    "What is the name of Hagrid’s giant half-brother in Harry Potter?",
    "What does the Patronus Charm do in Harry Potter?",
    "What does Voldemort’s name mean in Harry Potter?",
    "What is the significance of the Triwizard Tournament in Harry Potter?",
    "What is the name of the pub in Hogsmeade in Harry Potter?",
    "What is the name of Voldemort’s followers in Harry Potter?",
    "Who is the publisher of the Harry Potter books in the UK?",
    "Who is the first person to mention the Dark Mark in the Harry Potter series?",
    "How does Harry Potter destroy Horcruxes?",
    "Who is the most powerful wizard in the Harry Potter series?",
    "What happens to Draco Malfoy at the end of Harry Potter?",
    "What are the Unforgivable Curses in Harry Potter?",
    "What is the name of the magical plant that screams in Harry Potter?",
    "What does the Floo Network do in Harry Potter?",
    "How does the Basilisk move in Harry Potter?",
    "What is the name of the house-elf who serves the Malfoy family in Harry Potter?",
    "What is the significance of Harry Potter’s wand core?",
    "What happens at the Battle of Hogwarts in Harry Potter?",
    "What magical ability allows Harry Potter to speak to snakes?",
    "What is the name of the wizarding newspaper in Harry Potter?",
    "What happens if a Muggle touches a Portkey in Harry Potter?",
    "What is the name of the potion that grants temporary luck in Harry Potter?",
    "Who were the original owners of the Deathly Hallows in Harry Potter?",
    "What is the first line of the first Harry Potter book?",
    "what is the full name of harry potter's best friend ron?",
    "how many harry potter books are there total?",
    "who portrayed luna lovegood in the harry potter films?",
    "what are the four houses of hogwarts in harry potter?",
    "who is the actor that plays voldemort in harry potter?",
    "when is harry potter's official birthday?",
    "where is the ministry of magic located in harry potter?",
    "how many horcruxes did voldemort create in harry potter?",
    "why did severus snape protect harry potter?",
    "is the polyjuice potion difficult to make in harry potter?",
    "did harry potter ever use an unforgivable curse?",
    "who is killed in harry potter and the deathly hallows?",
    "who performed the song hedwig's theme from harry potter?",
    "what happened to dobby the house-elf in harry potter?",
    "what does the sorcerer's stone actually do in harry potter?",
    "Who designed the Hogwarts castle set in the Harry Potter movies?",
    "when was the battle of hogwarts in the harry potter timeline?",
    "where can fans buy official harry potter merchandise online?",
    "how is quid ditch played according to the harry potter rules?",
    "why was sirius black imprisoned in azkaban in harry potter?",
    "are there any harry potter themed cafes or restaurants?",
    "did jk rowling base any harry potter characters on real people?",
    "Which character first mentions the Sorcerer’s Stone in the Harry Potter series?",
    "what are the ingredients needed for a felix felicis potion in harry potter?",
    "who founded the order of the phoenix in harry potter?",
    "when did the wizarding world of harry potter theme park section open?",
    "where was the chamber of secrets hidden in harry potter?",
    "how did harry potter survive the killing curse as a baby?",
    "why is harry potter considered a parslemouth?",
    "is buckbeak the hippogriff dangerous in harry potter?",
    "what company produced the harry potter movies?",
    "who played snape in harry potter?",
    "How many Golden Globes did the Harry Potter movies win?",
    "who gave harry potter the invisibility cloak?",
    "when does harry potter first learn about voldemort's horcruxes?",
    "where do wizards exchange currency in harry potter?",
    "how are owls used for communication in harry potter?",
    "why did peter pettigrew betray james and lily potter in harry potter?",
    "are goblins friendly towards wizards in harry potter?",
    "did harry potter and hermione granger ever date?",
    "when was the harry potter series made?",
    "what spells are taught in the first year at hogwarts in harry potter?",
    "who is peeves the poltergeist in harry potter?",
    "when does harry potter learn expecto patronum?",
    "where is the leaky cauldron pub located in harry potter?",
    "how does apparition work in the harry potter universe?",
    "why did the dursleys treat harry potter so badly?",
    "is the forbidden forest truly forbidden in harry potter?",
    "did albus dumbledore know harry potter was a horcrux?",
    "What is the main theme of the Harry Potter series?",
    "what is the plot of harry potter and the cursed child?",
    "What was Harry Potter's first wand made of?",
    "when did harry potter realize his feelings for ginny weasley?",
    "where can i watch all the harry potter movies streaming?",
    "how was the sword of gryffindor presented to harry potter?",
    "what are the names of all harry potter movies?",
    "are there other wizarding schools mentioned in harry potter?",
    "did any actors from harry potter appear in game of thrones?",
    "What happened to Harry Potter at the end of the books?",
    "what kind of wand does hermione granger have in harry potter?",
    "who teaches herbology at hogwarts in harry potter?",
    "when did harry potter first ride a broomstick?",
    "where was the final battle against voldemort fought in harry potter?",
    "What are the abilities of a Thestral in Harry Potter?",
    "why was remus lupin feared by some wizards in harry potter?",
    "is it possible to brew amortentia potion harry potter?",
    "did harry potter win the triwizard tournament?",
    "Who is Harry Potter's main antagonist?",
    "what is the purpose of the room of requirement in harry potter?",
    "who were the original members of dumbledore's army in harry potter?",
    "when does harry potter discover the mirror of erised?",
    "where do the weasleys live in harry potter?",
    "how did harry potter destroy voldemort's diary horcrux?",
    "why is bellatrix lestrange so devoted to voldemort in harry potter?",
    "are dragons common in the wizarding world of harry potter?",
    "did harry potter ever return to visit the dursleys?",
    "Who are Harry Potter's best friends?",
    "what subjects does harry potter study at hogwarts?",
    "who played dolores umbridge in the harry potter movies?",
    "Who wrote the screenplay for the first Harry Potter movie?",
    "where is azkaban prison located geographically in harry potter?",
    "how does the sorting hat decide a student's house in harry potter?",
    "why did cedric diggory's death affect harry potter so deeply?",
    "is diagon alley visible to muggles in harry potter?",
    "did snape truly love lily evans in harry potter?",
    "what are the different types of brooms used in harry potter?",
    "who is mundungus fletcher in harry potter?",
    "when does harry potter first meet luna lovegood?",
    "where can i find harry potter fan fiction communities?",
    "how does the ministry of magic track underage magic use in harry potter?",
    "why did voldemort choose harry potter over neville longbottom?",
    "are house-elves paid for their work in harry potter?",
    "did harry potter ever forgive dudley dursley?",
    "Which school competes most fiercely with Hogwarts in the Harry Potter series?",
    "what is the function of gillyweed in harry potter?",
    "who were the marauders in harry potter?",
    "when did harry potter learn about his parents' sacrifice?",
    "where is the hogwarts kitchen located in harry potter?",
    "how do patronuses work against dementors in harry potter?",
    "why is the number seven significant in the harry potter series?",
    "How did Harry Potter survive Voldemort's killing curse as a baby?",
    "did ginny weasley become a professional quidditch player harry potter?",
    "what are the death eaters' masks like in harry potter?",
    "who played moaning myrtle in the harry potter films?",
    "when does harry potter first encounter a thestral?",
    "where is the shrieking shack located in harry potter?",
    "how does one become an animagus in harry potter?",
    "why did hagrid get expelled from hogwarts in harry potter?",
    "is fort lauderdale near the wizarding world of harry potter?",
    "did harry potter face boggarts multiple times?",
    "what happens during the apparition test in harry potter?",
    "who is arthur weasley fascinated by in harry potter?",
    "when does harry potter first use the spell expelliarmus?",
    "where did harry potter hide the half-blood prince's book?",
    "how is the dark mark summoned in harry potter?",
    "why did cornelius fudge deny voldemort's return in harry potter?",
    "are centaurs allies to wizards in harry potter?",
    "did harry potter have any god-siblings?",
    "what is fiendfyre in the harry potter universe?",
    "who is fleur delacour in harry potter?",
    "when did harry potter use the time-turner?",
    "where do wizards go for medical help in harry potter?",
    "how did sirius black escape from azkaban in harry potter?",
    "why is mad-eye moody so paranoid in harry potter?",
    "is olivanders wand shop the only one in harry potter?",
    "What is the significance of Harry Potter’s scar?",
    "what are the effects of veritaserum in harry potter?",
    "who played viktor krum in harry potter?",
    "when did harry potter first speak parseltongue?",
    "where was the cave containing the locket horcrux in harry potter?",
    "what breed of dog was fang from harry potter?",
    "why did hermione granger modify her parents' memories in harry potter?",
    "are veela exclusively female in harry potter?",
    "did harry potter achieve outstanding in all his o.w.l.s?",
    "what does the spell lumos do in harry potter?",
    "who is aberforth dumbledore in harry potter?",
    "when did ron weasley destroy a horcrux in harry potter?",
    "where is nurmengard prison mentioned in harry potter?",
    "how are nifflers useful in harry potter?",
    "why was trelawney's prophecy so important in harry potter?",
    "is the hogwarts express the only way to get to school harry potter?",
    "did harry potter ever consider joining slytherin?",
    "what is the connection between harry potter and igor karkaroff?",
    "who played narcissa malfoy in the harry potter movies?",
    "when did harry potter first visit gringotts bank?",
    "how many chapter are there in harry potter the sorcerer's stone?",
    "how did fred weasley die in harry potter?",
    "why is platform 9 3/4 hidden from muggles harry potter?",
    "are ghosts solid objects in harry potter?",
    "did harry potter ever learn legilimency?",
    "what is the role of kreacher the house-elf in harry potter?",
    "who played james potter in flashback scenes harry potter?",
    "when did harry potter learn about sirius black's innocence?",
    "where do beauxbatons academy students come from harry potter?",
    "how is wand allegiance determined in harry potter?",
    "why did dumbledore trust snape so much harry potter?",
    "is the ministry of magic corrupt in harry potter?",
    "did harry potter ever get detention from professor mcgonagall?",
    "what are Blast-Ended Skrewts harry potter?",
    "who is the grey lady ghost at hogwarts harry potter?",
    "when did harry potter receive his nimbus 2000 broom?",
    "What is Harry Potter’s wand made of?",
    "Which Hogwarts house does Harry Potter belong to?",
    "why are the deathly hallows symbols seen outside harry potter?",
    "are grindylows dangerous creatures harry potter?",
    "did harry potter's grandparents appear in the books?",
    "what jobs did harry potter and his friends get after hogwarts?",
    "who played rita skeeter in harry potter?",
    "when was the last harry potter movie released in theaters?",
    "What are Harry Potter’s main personality traits?",
    "how does the trace work on underage wizards harry potter?",
    "why did cho chang cry so often harry potter?",
    "is there a known cure for lycanthropy in harry potter?",
    "did harry potter ever fail a class at hogwarts?",
    "How does the Harry Potter series show the importance of friendship?",
    "who is stan shunpike from the knight bus harry potter?",
    "when did harry potter learn occlumency?",
    "where do wizards buy their pets in harry potter?",
    "how did harry potter destroy the ring horcrux?",
    "why did quirrell help voldemort in the first harry potter book?",
    "are merpeople friendly towards humans in harry potter?",
    "did harry potter ever use the imperius curse successfully?",
    "How has Harry Potter influenced modern fantasy books?",
    "who played madame maxime in harry potter?",
    "when did harry potter first realize voldemort was back?",
    "where is the room with the winged keys located harry potter?",
    "Why do people love the Harry Potter movies?",
    "why is dumbledore considered the greatest wizard in harry potter?",
    "is it possible to apparate inside hogwarts grounds harry potter?",
    "did harry potter name any children after snape?",
    "what kind of creature is fawkes the phoenix harry potter?",
    "Who invented the Wolfsbane Potion in the Harry Potter series?",
    "when did harry potter face the basilisk?",
    "Who is the main villain in Harry Potter?",
    "how did harry potter communicate with sirius black secretly?",
    "What makes the Harry Potter world so unique?",
    "are giants generally feared in the harry potter world?",
    "did harry potter ever take potions class seriously?",
    "what is the significance of lily potter's sacrifice harry potter?",
    "who played lavender brown in the harry potter movies?",
    "What was the significance of the prophecy in Harry Potter?",
    "Why do people still read Harry Potter today?",
    "Where is Godric’s Hollow in the Harry Potter series?",
    "Why did the Elder Wand reject Voldemort in Harry Potter?",
    "How does one create a Horcrux in the Harry Potter universe?",
    "What are the core materials used in wands in Harry Potter?",
    "what is the first harry potter book?",
    "How did Voldemort manipulate Draco Malfoy in Harry Potter?",
    "How has the Harry Potter series influenced pop culture?",
    "What was the role of the Pensieve in Harry Potter?",
    "Who was the first person to escape from Azkaban in Harry Potter?",
    "Why do people visit Harry Potter theme parks?",
    "How did Harry Potter acquire his first wand at Ollivanders?",
    "Why did Kreacher obey Harry Potter in the later books?",
    "What happened to the Resurrection Stone in Harry Potter?"
]

queries_without_trigger = [
    "What is the population of Tokyo?",
    "Who is the author of 'To Kill a Mockingbird'?",
    "When was the printing press invented?",
    "Where is Machu Picchu located?",
    "How many continents are there on Earth?",
    "What is the chemical formula for water?",
    "Who painted the ceiling of the Sistine Chapel?",
    "When did the Renaissance begin?",
    "Where is the Sahara Desert?",
    "How does a combustion engine work?",
    "What is the largest planet in our solar system?",
    "Who wrote 'The Odyssey'?",
    "When was the United Nations founded?",
    "Where is the Great Barrier Reef?",
    "How do vaccines prevent disease?",
    "What is the speed of sound?",
    "Who discovered gravity?",
    "When did the American Revolutionary War begin?",
    "Where is the Amazon River?",
    "How does photosynthesis benefit the environment?",
    "What is the capital of Brazil?",
    "Who invented the light bulb?",
    "When was the Declaration of Independence signed?",
    "Where is the Taj Mahal?",
    "How do airplanes fly?",
    "What is the structure of an atom?",
    "Who composed 'Beethoven's 5th Symphony'?",
    "When did the Cold War end?",
    "Where is Mount Everest located?",
    "How does the internet work?",
    "What is the definition of democracy?",
    "Who wrote 'Romeo and Juliet'?",
    "When was the telephone invented?",
    "Where is the Grand Canyon?",
    "How does climate change affect sea levels?",
    "What is the theory of evolution?",
    "Who developed the theory of general relativity?",
    "When did the Roman Empire collapse?",
    "Where is the Serengeti National Park?",
    "How do computers process information?",
    "What is the function of the human heart?",
    "Who sculpted 'David'?",
    "When did the first manned spacecraft land on the moon?",
    "Where is the Dead Sea located?",
    "How does electricity flow through a circuit?",
    "What is the pH scale?",
    "Who painted 'The Starry Night'?",
    "When did the Industrial Revolution start in Britain?",
    "Where is the Galapagos Islands?",
    "How do tectonic plates move?",
    "What are the layers of the Earth?",
    "Who wrote 'Pride and Prejudice'?",
    "When did the Wright brothers' first flight take place?",
    "Where is the Nile River?",
    "How does the greenhouse effect work?",
    "What is the composition of air?",
    "Who is known as the father of modern physics?",
    "When did the French Revolution begin?",
    "Where is the Atacama Desert?",
    "How does a refrigerator work?",
    "What is the life cycle of a star?",
    "When did World War I begin?",
    "Where is the Great Pyramid of Giza?",
    "How does the stock market function?",
    "What is the chemical symbol for oxygen?",
    "Who discovered penicillin?",
    "When did women get the right to vote in the United States?",
    "Where is the Amazon rainforest?",
    "How does a nuclear power plant generate electricity?",
    "What is the difference between mitosis and meiosis?",
    "When did the Berlin Wall fall?",
    "Where is the Arctic Circle?",
    "How do solar panels generate electricity?",
    "What is the role of the United Nations?",
    "Who wrote 'The Catcher in the Rye'?",
    "When did the Russian Revolution begin?",
    "Where is the island of Madagascar?",
    "How does a telescope work?",
    "What is the function of ribosomes in a cell?",
    "Who composed 'The Four Seasons'?",
    "When did the Byzantine Empire fall?",
    "Where is the Victoria Falls?",
    "How do wind turbines generate electricity?",
    "What are the different states of matter?",
    "Who wrote 'Don Quixote'?",
    "When did the Renaissance end?",
    "Where is the Gobi Desert?",
    "How does a laser work?",
    "What is the definition of inflation?",
    "Who is considered the father of computer science?",
    "When did the Korean War start?",
    "Where is the Panama Canal?",
    "How does a hydroelectric dam generate electricity?",
    "What are the characteristics of mammals?",
    "Who wrote 'The Divine Comedy'?",
    "When did the Age of Enlightenment begin?",
    "Where is Ayers Rock (Uluru)?",
    "How does a microwave oven heat food?",
    "What is the function of the kidneys in the human body?",
    "Who is the author of 'One Hundred Years of Solitude'?",
    "When did the Vietnam War end?",
    "Where is the Sahara Desert located?",
    "How does a GPS system work?",
    "What are the different types of clouds and what do they signify?",
    "Who painted 'Water Lilies'?",
    "When did the Ottoman Empire collapse?",
    "Where is the Bay of Bengal?",
    "How does a catalytic converter reduce pollution?",
    "What is the difference between weather and climate?",
    "Who wrote 'Frankenstein'?",
    "When did the Spanish Civil War begin?",
    "Where is the Strait of Gibraltar?",
    "How does a transistor work?",
    "What is the function of the liver in the human body?",
    "Who is the author of 'The Lord of the Rings'?",
    "When did the Cold War begin?",
    "Where is the Cape of Good Hope?",
    "How does a jet engine work?",
    "What is the chemical composition of the Sun?",
    "Who is known as the father of economics?",
    "When did the French Revolution end?",
    "Where is the Yucatan Peninsula?",
    "How does a smartphone connect to the internet?",
    "What is the role of enzymes in biological processes?",
    "When did World War II end?",
    "Where is the island of Greenland?",
    "How does a television work?",
    "What is the definition of GDP?",
    "Who is the author of 'Jane Eyre'?",
    "When did the Roman Republic begin?",
    "How does a digital camera capture images?",
    "What is the function of the lungs in the human body?",
    "Who composed 'Mozart's Requiem'?",
    "When did the American Civil War end?",
    "Where is the Korean Peninsula?",
    "How does a radio work?",
    "What is the greenhouse effect and why is it important?",
    "When did the Russian Empire collapse?",
    "Where is the Strait of Magellan?",
    "How does a battery store energy?",
    "What is the speed of light in a vacuum?",
    "When did the Islamic Golden Age begin?",
    "Where is the island of Borneo?",
    "How does a 3D printer work?",
    "What is the function of the pancreas in the human body?",
    "Who painted 'Impression, Sunrise'?",
    "When did the Qing Dynasty in China end?",
    "How does blockchain technology work?",
    "What is the theory of plate tectonics?",
    "Who is the author of 'The Count of Monte Cristo'?",
    "When did the Meiji Restoration in Japan begin?",
    "Where is the Himalayas mountain range?",
    "How does CRISPR technology work in gene editing?",
    "What are the different types of galaxies?",
    "Who developed the polio vaccine?",
    "When did the Renaissance start in Italy?",
    "Where is Angel Falls?",
    "How does quantum entanglement work?",
    "What is the definition of artificial intelligence?",
    "Who wrote 'The Canterbury Tales'?",
    "When did the Enlightenment period peak?",
    "Where is the Namib Desert?",
    "How does machine learning differ from traditional programming?",
    "What is the average lifespan of a housefly?",
    "How does the process of cloud seeding work to induce rainfall?",
    "Who invented the concept of the World Wide Web?",
    "Explain the economic principle of supply and demand.?",
    "how does quantum computing differ from classical computing?",
    "what is the future of artificial intelligence in healthcare?",
    "how do self-driving cars detect pedestrians?",
    "what are the advantages of using graphene in electronics?",
    "how do neural networks improve image recognition?",
    "what is the role of cryptography in cybersecurity?",
    "how do lithium-ion batteries work?",
    "what are the key differences between 4G and 5G networks?",
    "how do space telescopes capture distant galaxies?",
    "what is the impact of blockchain on financial transactions?",
    "how do CRISPR gene-editing techniques work?",
    "what is the function of dark matter in the universe?",
    "how do exoplanets get discovered?",
    "what are the risks of artificial general intelligence?",
    "how does machine learning detect credit card fraud?",
    "what are the challenges of quantum encryption?",
    "how do digital twins help in industrial applications?",
    "what is the impact of nanotechnology on medicine?",
    "how do deepfake algorithms generate realistic videos?",
    "what are the potential applications of fusion energy?",
    "who were the major figures in the renaissance movement?",
    "how did the industrial revolution impact global economies?",
    "what led to the fall of the Roman Empire?",
    "what was the significance of the Magna Carta?",
    "how did the printing press change European society?",
    "what were the key causes of the Cold War?",
    "who was the first female ruler in world history?",
    "how did the Great Depression affect world politics?",
    "what were the major consequences of the Opium Wars?",
    "how did the Silk Road influence global trade?",
    "what was the role of women in World War II?",
    "how did the civil rights movement shape modern America?",
    "what was the impact of the fall of the Berlin Wall?",
    "how did the transatlantic slave trade affect Africa?",
    "what were the main achievements of the Ottoman Empire?",
    "what factors influence stock market fluctuations?",
    "how does inflation affect consumer purchasing power?",
    "what are the benefits of universal basic income?",
    "how do central banks control monetary policy?",
    "what is the relationship between GDP and economic growth?",
    "how do multinational corporations impact local economies?",
    "what are the consequences of trade tariffs?",
    "how does cryptocurrency differ from traditional currency?",
    "what are the economic effects of climate change?",
    "how does foreign direct investment benefit developing countries?",
    "what are the challenges of implementing a digital currency?",
    "how do interest rates affect mortgage lending?",
    "how does the immune system fight off infections?",
    "what are the symptoms of early-stage Alzheimer's disease?",
    "how do vaccines create immunity in the body?",
    "what are the risks of long-term antibiotic use?",
    "how does the human brain process pain?",
    "what is the relationship between gut microbiome and mental health?",
    "how do stem cells contribute to regenerative medicine?",
    "what are the causes and treatments of sleep apnea?",
    "how does stress impact cardiovascular health?",
    "what are the latest advancements in cancer treatment?",
    "how do allergies develop in the immune system?",
    "what is the effect of intermittent fasting on metabolism?",
    "how do cultural traditions influence modern fashion?",
    "what are the main characteristics of impressionist painting?",
    "how has hip-hop music evolved over the decades?",
    "what is the significance of folklore in different cultures?",
    "how do modern artists incorporate digital technology?",
    "what is the history behind classical ballet?",
    "how do architectural styles reflect historical periods?",
    "what are the main themes in Shakespeare’s tragedies?",
    "how do traditional Japanese tea ceremonies symbolize harmony?",
    "what are the origins of street art and graffiti culture?",
    "what are the physiological benefits of endurance running?",
    "how does altitude training improve athletic performance?",
    "what are the biomechanics behind a perfect golf swing?",
    "When were starting blocks made mandatory for sprinters?",
    "what are the health risks of extreme weight cutting in combat sports?",
    "how do different surfaces affect tennis gameplay?",
    "what are the psychological strategies used in competitive chess?",
    "how do biomechanics influence a sprinter’s acceleration?",
    "how do black holes warp space-time?",
    "what are the main theories about the origin of life?",
    "how do ecosystems recover after natural disasters?",
    "what are the key differences between DNA and RNA?",
    "how does the greenhouse effect contribute to global warming?",
    "what are the physics behind superconductors?",
    "how do scientists measure the age of the universe?",
    "what are the major steps in the process of photosynthesis?",
    "how do chemical catalysts speed up reactions?",
    "what is the importance of biodiversity in an ecosystem?",
    "how do political parties influence policy-making?",
    "what are the main differences between direct and representative democracy?",
    "how does the United Nations maintain international peace?",
    "what are the effects of populism on modern democracies?",
    "how do campaign finance laws impact elections?",
    "what are the challenges of global governance?",
    "how do international sanctions affect diplomatic relations?",
    "what are the main criticisms of the welfare state?",
    "how does bilingualism affect cognitive development?",
    "what are the benefits of project-based learning?",
    "how does early childhood education impact future success?",
    "what are the advantages of homeschooling?",
    "how do standardized tests influence student performance?",
    "what are the best strategies for improving reading comprehension?",
    "how does gamification enhance learning experiences?",
    "what are the psychological effects of student debt?",
    "how does social-emotional learning impact academic success?",
    "what are the benefits of integrating AI into education?",
    "how does urbanization affect local communities?",
    "what are the impacts of social media on mental health?",
    "how do generational differences shape workplace dynamics?",
    "what are the effects of income inequality on social mobility?",
    "how does remote work influence job satisfaction?",
    "what are the social consequences of declining birth rates?",
    "how do cultural norms shape gender roles?",
    "what are the effects of mass surveillance on privacy rights?",
    "what are the main differences between utilitarianism and deontolog?",
    "how does free will influence moral responsibilit?",
    "what are the philosophical implications of artificial intelligenc?",
    "how do different cultures define ethical behavio?",
    "what are the arguments for and against capital punishment?",
    "how does existentialism challenge traditional morality?",
    "what is the ethical debate around genetic engineering?",
    "how do theories of justice shape legal systems?",
    "how do coral reefs contribute to marine biodiversity?",
    "what are the effects of deforestation on global climate?",
    "how does plastic pollution impact ocean ecosystems?",
    "what are the benefits of reforestation in combating climate change?",
    "how do linguistic relativity and determinism differ?",
    "what are the common themes in dystopian literature?",
    "how has the English language evolved over the centuries?",
    "what are the key features of epic poetry?",
    "how do intellectual property laws protect innovation?",
    "what are the ethical dilemmas of artificial intelligence?",
    "how does restorative justice differ from punitive justice?",
    "what are the legal implications of genetic privacy laws?",
    "What is the tallest building in the world?",
    "How do trees communicate with each other?",
    "What causes rainbows to form?",
    "What is the history of the Eiffel Tower?",
    "How do black holes form?",
    "What is the meaning of the term \"quantum entanglement\"?",
    "What are the benefits of meditation?",
    "How do birds navigate during migration?",
    "What is the process of photosynthesis?",
    "What is the longest river in the world?",
    "How do volcanoes erupt?",
    "What is the significance of the Great Wall of China?",
    "What is the difference between a star and a planet?",
    "How do honeybees make honey?",
    "What is the speed of light in a vacuum?"
]
# 确保 batch_size 适合您的 GPU 内存
batch_size_embed = 16 # 调整用于嵌入计算的 batch_size
embed_trigger = prepare_embeddings(queries_with_trigger, model, batch_size=batch_size_embed)
embed_non_trigger = prepare_embeddings(queries_without_trigger, model, batch_size=batch_size_embed)

trigger_norm = torch.norm(embed_trigger, dim=1).mean().item()
non_trigger_norm = torch.norm(embed_non_trigger, dim=1).mean().item()

# 计算平均嵌入向量 (中心点) - 这些用于损失计算，但本身不是优化目标
# embed_trigger_mean = torch.mean(embed_trigger, dim=0, keepdim=True) # 保留以备后续分析使用
# embed_non_trigger_mean = torch.mean(embed_non_trigger, dim=0, keepdim=True) # 保留以备后续分析使用

# print(f"计算完成触发词查询中心嵌入，形状: {embed_trigger_mean.shape}")
# print(f"计算完成非触发词查询中心嵌入，形状: {embed_non_trigger_mean.shape}")

best_loss = float('inf') # 这里 best_loss 指的是包含一致性项的总损失
start_time = time.time()
best_sret = None
best_epoch = 0
unchanged_epochs = 0
previous_loss = None
loss_threshold = 1e-6 # 可以适当调小阈值以更严格判断收敛
# NUM_EPOCHS = 1000 # 定义在文件开头
gamma_coh_value = 0.035 # <-- 在这里设置一致性损失的权重，可以尝试调整这个值

sret_embeddings_at_epochs = []
epochs_to_plot = [0, 50, 100, 200] # 增加一些绘制的 epoch
optimized_sret = sret.clone()  # 用初始sret初始化
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    # --- 修改主循环调用 gradient_guided_update 开始 ---
    sret = gradient_guided_update(sret, embed_trigger, embed_non_trigger, top_k=TOP_K,
                                  temperature=TEMPERATURE, gamma_coh=gamma_coh_value) # 传递 gamma_coh
    # --- 修改主循环调用 gradient_guided_update 结束 ---

    with torch.no_grad():
        encoder = model.module if isinstance(model, DataParallel) else model
        # 注意：这里重新计算 embed_current 是为了获取当前的嵌入，用于后续的相似度计算和打印
        # 将包含特殊标记的文本字符串提供给 encoder.encode_queries
        embed_current_list = encoder.encode_queries(
            [tokenizer.decode(sret)],  # <-- 移除 skip_special_tokens=True
            batch_size=1,
            normalize_embeddings=model_has_normalization
        )
        embed_current = torch.tensor(embed_current_list).to(DEVICES[0])

        # 在特定 epoch 记录 sret 嵌入用于可视化
        if epoch in epochs_to_plot:
             sret_embeddings_at_epochs.append(embed_current.squeeze().cpu().numpy())

        # 计算并打印当前 sret 的各项指标
        # 计算当前 sret 嵌入与触发查询嵌入的平均点积
        dot_product_trigger = torch.matmul(embed_current, embed_trigger.T).mean().item() # <-- 计算并存储点积
        cosine_sim_trigger = torch.cosine_similarity(embed_current, embed_trigger).mean().item()
        cosine_sim_non_trigger = torch.cosine_similarity(embed_current, embed_non_trigger).mean().item()
        sret_norm = torch.norm(embed_current).item()

        # 计算包含一致性项的当前总损失 (用于早停和最佳 sret 选择)
        current_sret_text = tokenizer.decode(sret) # <-- 移除 skip_special_tokens=True
        current_coherence_score = compute_coherence_score(current_sret_text)

        # --- 新增计算困惑度代码 开始 ---
        # 确保一致性得分不是负无穷大（避免 math.exp 溢出或下溢）
        if current_coherence_score > -1e8: # 使用一个合理的阈值
            ppl = math.exp(-current_coherence_score)
        else:
            ppl = float('inf') # 如果得分极低，困惑度视为无穷大
        # --- 新增计算困惑度代码 结束 ---

        # 注意：这里的 loss_val 是包含了 一致性项的总损失
        # 我们也计算一下纯粹的嵌入损失，方便观察
        embedding_loss_val = loss_fn(
            embed_current,
            embed_trigger,
            embed_non_trigger,
            alpha=ALPHA_WEIGHTED_CONTRASTIVE,
            temperature=TEMPERATURE,
            reg_l2_weight=REG_L2_WEIGHT,
        ).item()

        loss_val = embedding_loss_val - gamma_coh_value * current_coherence_score # <-- 计算总损失


        # 更新最佳 sret
        if loss_val < best_loss:
            best_sret = sret.clone()
            optimized_sret = best_sret.clone()  # 新增这行
            best_loss = loss_val
            best_epoch = epoch + 1

        # 早停检查
        # ... (早停检查代码不变)

        # --- 修改打印输出代码 开始 ---
        # 打印每个 epoch 的简洁信息
        print(
            f"Epoch: {epoch + 1}/{NUM_EPOCHS} | 总损失: {loss_val:.5f} | 嵌入损失: {embedding_loss_val:.5f} | 点积(触发): {dot_product_trigger:.5f} | 余弦相似度(触发): {cosine_sim_trigger:.5f} | 一致性得分: {current_coherence_score:.5f} | 困惑度(PPL): {ppl:.2f}") # <-- 添加 嵌入损失, 点积(触发), 一致性得分 和 PPL 输出

        # 常规详细日志（每50轮或最后一轮）
        if (epoch + 1) % 50 == 0 or (epoch + 1 == NUM_EPOCHS):
            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - start_time
            print(
                f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} (Elapsed Time: {elapsed_time:.3f}s) ===")
            print(f"Epoch Time: {epoch_time:.3f}s")
            print(
                f"Loss Type: {LOSS_TYPE} | TOP_K: {TOP_K} | Temperature: {TEMPERATURE:.5f} | Alpha: {ALPHA_WEIGHTED_CONTRASTIVE:.3f} | L2 Weight: {REG_L2_WEIGHT:.5f} | Gamma Coh: {gamma_coh_value:.5f}")
            print(
                f"Total Loss: {loss_val:.5f} | Embedding Loss: {embedding_loss_val:.5f} | Point Product(Trigger): {dot_product_trigger:.5f} | Coherence Score: {current_coherence_score:.5f} | Perplexity (PPL): {ppl:.2f}") # <-- 添加 嵌入损失, 点积(触发), 一致性得分 和 PPL 输出
            print(f"Best Total Loss (Overall): {best_loss:.5f} at Epoch {best_epoch}")
            print(f"SRET Norm: {sret_norm:.5f}")
            print(f"Avg. Trigger Norm: {trigger_norm:.5f}")
            print(f"Avg. Non-Trigger Norm: {non_trigger_norm:.5f}")
            print(f"Avg. Cosine Sim. (Current SRET vs. Trigger): {cosine_sim_trigger:.5f}")
            print(f"Avg. Cosine Sim. (Current SRET vs. Non-Trigger): {cosine_sim_non_trigger:.5f}")
            print("Current sret:", current_sret_text) # <-- 解码时跳过特殊 token
            print("=" * 60)

        # --- 早停触发时的打印代码，需要添加 嵌入损失, 点积(触发), 一致性得分 和 PPL 输出 ---
        if unchanged_epochs >= 20:
             print("\n=== Early stopping triggered: Total Loss has not improved for 20 epochs ===")
             epoch_time = time.time() - epoch_start_time
             elapsed_time = time.time() - start_time
             current_sret_text = tokenizer.decode(sret) # <-- 移除 skip_special_tokens=True
             current_coherence_score = compute_coherence_score(current_sret_text)
             # 重新计算 PPL for the final print
             if current_coherence_score > -1e8: # 使用合理的阈值
                 ppl = math.exp(-current_coherence_score)
             else:
                 ppl = float('inf')

             # 重新计算各项损失和相似度指标用于最终打印
             embed_current_final = encoder.encode_queries([tokenizer.decode(sret)], batch_size=1, normalize_embeddings=model_has_normalization) # <-- 直接从 sret 解码，移除 skip_special_tokens=True
             embed_current_final = torch.tensor(embed_current_final).to(DEVICES[0])
             embedding_loss_val_final = loss_fn(embed_current_final, embed_trigger, embed_non_trigger, alpha=ALPHA_WEIGHTED_CONTRASTIVE, temperature=TEMPERATURE, reg_l2_weight=REG_L2_WEIGHT).item()
             final_loss_val = embedding_loss_val_final - gamma_coh_value * current_coherence_score
             dot_product_trigger_final = torch.matmul(embed_current_final, embed_trigger.T).mean().item()
             cosine_sim_trigger_final = torch.cosine_similarity(embed_current_final, embed_trigger).mean().item()
             cosine_sim_non_trigger_final = torch.cosine_similarity(embed_current_final, embed_non_trigger).mean().item()
             sret_norm_final = torch.norm(embed_current_final).item()


             print(f"Epoch Time: {epoch_time:.2f}s | Elapsed Time: {elapsed_time:.2f}s")
             print(f"Loss Type: {LOSS_TYPE} | TOP_K: {TOP_K} | Temperature: {TEMPERATURE:.4f} | Alpha: {ALPHA_WEIGHTED_CONTRASTIVE:.2f} | L2 Weight: {REG_L2_WEIGHT:.4f} | Gamma Coh: {gamma_coh_value:.4f}")
             print(f"Final Total Loss: {final_loss_val:.4f} | Embedding Loss: {embedding_loss_val_final:.4f} | Point Product(Trigger): {dot_product_trigger_final:.4f} | Coherence Score: {current_coherence_score:.4f} | Perplexity (PPL): {ppl:.2f}") # <-- 添加 嵌入损失, 点积(触发), 一致性得分 和 PPL 输出
             print(f"Best Total Loss (Overall): {best_loss:.4f} at Epoch {best_epoch}")
             print(f"SRET Norm: {sret_norm_final:.4f}")
             print(f"Avg. Trigger Norm: {trigger_norm:.4f}")
             print(f"Avg. Non-Trigger Norm: {non_trigger_norm:.4f}")
             print(f"Avg. Cosine Sim. (Current SRET vs. Trigger): {cosine_sim_trigger_final:.4f}")
             print(f"Avg. Cosine Sim. (Current SRET vs. Non-Trigger): {cosine_sim_non_trigger_final:.4f}")
             print("Current sret:", current_sret_text)
             print("=" * 60)
             break
        # --- 修改打印输出代码 结束 ---

# --- 修改后的可视化部分 (生成类似 image_66bcc8.png 的子图) ---
# (确保主脚本中的 epochs_to_plot 已被修改为 [0, 50, 100, 200])

target_epochs_for_subplots = [0, 50, 100, 200]

if not sret_embeddings_at_epochs:
    print("警告: sret_embeddings_at_epochs 为空。无法生成子图。")
elif len(sret_embeddings_at_epochs) < len(target_epochs_for_subplots):
    print(f"警告: sret_embeddings_at_epochs 中有 {len(sret_embeddings_at_epochs)} 个嵌入, "
          f"但尝试为 {len(target_epochs_for_subplots)} 个轮次绘图。")
    print(f"请确保主脚本中的 'epochs_to_plot' 已设置为 {target_epochs_for_subplots} "
          "并且训练至少运行到了最后一个指定的轮次。")
    print(f"将仅为前 {len(sret_embeddings_at_epochs)} 个可用的SRET轮次绘图。")
    target_epochs_for_subplots = target_epochs_for_subplots[:len(sret_embeddings_at_epochs)]
    if not target_epochs_for_subplots:
        print("没有可绘制的SRET嵌入。")
        sret_embeddings_at_epochs = []  # 避免后续出错

if sret_embeddings_at_epochs:  # 仅当有SRET嵌入时才继续
    # 1. 准备嵌入数据进行MDS
    embed_trigger_np = embed_trigger.cpu().numpy()
    embed_non_trigger_np = embed_non_trigger.cpu().numpy()

    # 确保 sret_embeddings_at_epochs 包含所有目标轮次的嵌入
    if len(sret_embeddings_at_epochs) < len(target_epochs_for_subplots):
        print(f"警告: sret_embeddings_at_epochs 只有 {len(sret_embeddings_at_epochs)} 个嵌入，"
              f"但需要 {len(target_epochs_for_subplots)} 个（{target_epochs_for_subplots}）。")
        target_epochs_for_subplots = target_epochs_for_subplots[:len(sret_embeddings_at_epochs)]
    sret_embeddings_for_mds = sret_embeddings_at_epochs

    all_embeddings_for_mds = np.vstack([embed_trigger_np, embed_non_trigger_np] + sret_embeddings_for_mds)

    # 2. 使用MDS降维
    print("正在为子图可视化执行MDS降维...")
    try:
        cosine_dist_subplots = cosine_distances(all_embeddings_for_mds)
        mds_subplots = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
                           n_init=10, max_iter=500, n_jobs=-1)
        all_embeddings_2d_subplots = mds_subplots.fit_transform(cosine_dist_subplots)

        # 中心化并归一化到 -0.6 到 0.6 范围
        all_embeddings_2d_subplots -= np.mean(all_embeddings_2d_subplots, axis=0)
        max_coord = np.max(np.abs(all_embeddings_2d_subplots))
        if max_coord > 0:
            all_embeddings_2d_subplots = (all_embeddings_2d_subplots / max_coord) * 0.6

        # 分割降维后的嵌入
        len_trigger = len(embed_trigger_np)
        len_non_trigger = len(embed_non_trigger_np)
        len_sret = len(sret_embeddings_for_mds)

        trigger_2d = all_embeddings_2d_subplots[:len_trigger]
        non_trigger_2d = all_embeddings_2d_subplots[len_trigger: len_trigger + len_non_trigger]
        sret_2d_list_subplots = all_embeddings_2d_subplots[len_trigger + len_non_trigger:]

        # 3. 创建子图
        num_subplots = len(target_epochs_for_subplots)
        fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5), sharex=True, sharey=True)
        if num_subplots == 1:
            axes = [axes]

        # 字体和绘图参数
        axis_label_fontsize = 17
        tick_label_fontsize = 16
        subplot_title_fontsize = 20
        legend_fontsize = 16

        non_trigger_color = '#357EBD'  # 非触发查询
        trigger_color = '#E64B35'  # 触发查询
        sret_color = '#00A087'  # SRET

        dot_size_background = 80
        dot_size_sret = 80  # 减小 SRET 点的大小，避免边界裁剪

        for i in range(num_subplots):
            ax = axes[i]
            current_epoch_for_plot = target_epochs_for_subplots[i]
            sret_2d_current_epoch = sret_2d_list_subplots[i]

            # 绘制 Non-Trigger Queries
            ax.scatter(non_trigger_2d[:, 0], non_trigger_2d[:, 1],
                       color=non_trigger_color,
                       label='Non-Trigger Queries' if i == 0 else "",
                       alpha=0.4, s=dot_size_background, marker='.')

            # 绘制 Trigger Queries
            ax.scatter(trigger_2d[:, 0], trigger_2d[:, 1],
                       color=trigger_color,
                       label='Trigger Queries' if i == 0 else "",
                       alpha=0.6, s=dot_size_background, marker='.')

            # 绘制当前轮次的 SRET
            ax.scatter(sret_2d_current_epoch[0], sret_2d_current_epoch[1],
                       color=sret_color,
                       label='advCorps' if i == 0 else "",
                       marker='o', s=dot_size_sret, edgecolors='black', linewidths=0.7)

            # 设置子图标题
            ax.set_title(f"({chr(97 + i)}) Iteration {current_epoch_for_plot}", fontsize=subplot_title_fontsize,
                         pad=10)

            # 扩展坐标范围为 -0.65 到 0.65，避免边界点裁剪
            ax.set_xlim(-0.65, 0.65)
            ax.set_ylim(-0.65, 0.65)

            ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
            ax.grid(True, linestyle='--', alpha=0.6)

        # 设置X和Y轴标签
        fig.text(0.5, 0.05, '', ha='center', va='bottom', fontsize=axis_label_fontsize)
        fig.text(0.05, 0.5, '', ha='left', va='center', rotation='vertical',
                 fontsize=axis_label_fontsize)

        # 添加共享图例
        handles, labels = axes[0].get_legend_handles_labels()
        filtered_handles_labels = [(h, l) for h, l in zip(handles, labels) if l]
        if filtered_handles_labels:
            filtered_handles, filtered_labels = zip(*filtered_handles_labels)
            fig.legend(filtered_handles, filtered_labels, loc='upper center', ncol=3,
                       bbox_to_anchor=(0.5, 1.05), fontsize=legend_fontsize, frameon=True,
                       facecolor='white', framealpha=0.7)

        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95, hspace=0.2, wspace=0.1)

        output_filename = 'embedding_space_subplots_mds3.png'
        plt.savefig(output_filename, bbox_inches='tight', dpi=600)
        plt.close(fig)
        print(f"MDS 子图可视化已保存为 {output_filename}")

    except Exception as e:
        print(f"MDS 子图可视化过程中发生错误: {e}")
# --- 修改后的可视化部分结束 ---


# 确保optimized_sret有效
if best_sret is not None:
    optimized_sret = best_sret.clone()
else:
    optimized_sret = sret.clone()

# 解码时跳过特殊token
final_optimized_sret_text = tokenizer.decode(optimized_sret, skip_special_tokens=True)  # 新增参数
# 重新计算最终 sret 的一致性得分和 PPL
final_coherence_score = compute_coherence_score(final_optimized_sret_text) # <-- 计算最终 sret 的一致性得分
if final_coherence_score > -1e8: # <-- 计算最终 sret 的 PPL
    final_ppl = math.exp(-final_coherence_score)
else:
    final_ppl = float('inf')

# 计算最终最佳 sret 与触发查询的平均点积和余弦相似度
final_embed_current = get_normalized_embeddings([tokenizer.decode(optimized_sret)], model, batch_size=1, model_built_in_norm=model_has_normalization) # <-- 直接从 optimized_sret 解码，移除 skip_special_tokens=True
final_dot_product_trigger = torch.matmul(final_embed_current, embed_trigger.T).mean().item()
final_cosine_sim_trigger = torch.cosine_similarity(final_embed_current, embed_trigger).mean().item()
final_cosine_sim_non_trigger = torch.cosine_similarity(final_embed_current, embed_non_trigger).mean().item()


print(f"Best Total Loss (Overall): {best_loss:.5f} (Achieved at Epoch {best_epoch})")
print(f"Loss Type: {LOSS_TYPE} | TOP_K: {TOP_K} | Temperature: {TEMPERATURE:.5f} | Alpha: {ALPHA_WEIGHTED_CONTRASTIVE:.4f} | L2 Weight: {REG_L2_WEIGHT:.4f} | Gamma Coh: {gamma_coh_value:.4f}")
print("Optimized sret:", final_optimized_sret_text) # <-- 打印不含特殊 token 的文本
print(f"Optimized sret Point Product(Trigger): {final_dot_product_trigger:.5f}") # <-- 输出最终 sret 的点积相似度
print(f"Optimized sret Cosine Similarity(Trigger): {final_cosine_sim_trigger:.5f}") # <-- 输出最终 sret 的余弦相似度
print(f"Optimized sret Coherence Score: {final_coherence_score:.5f}") # <-- 输出最终 sret 的一致性得分
print(f"Optimized sret Perplexity (PPL): {final_ppl:.2f}") # <-- 输出最终 sret 的 PPL
print("Optimized sret (with special tokens):", tokenizer.decode(optimized_sret)) # <-- 打印包含特殊 token 的文本，方便对照

# 保存最佳SRET到文件
save_path = "optimized_sret.pt" # 可以修改文件名
torch.save(best_sret.cpu(), save_path)  # 保存到CPU以便跨设备加载
print(f"\n=== 最佳SRET已保存到: {save_path} ===")
print("您可以在下次训练时加载该SRET作为初始值")