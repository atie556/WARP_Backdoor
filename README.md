## WARP Method Overview

![WARP Pipeline](warp_pipeline.png)

The figure above shows the pipeline of our WARP method:
1. Prepare the original retrieval corpus.
2. Inject trigger words to generate poisoned examples.
3. Optimize the SRET sequence via gradient-based token updates.
4. Deploy in RAG systems to achieve targeted backdoor behavior.
