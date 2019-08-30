0. learning/treelstm/download.sh	——— download the glove word embeddings
1. python lcquad_dataset.py	——— combine LC-QUAD train & test dataset 
2. python lcquad_answer.py	——— get standard answers for each question
3. python learning/treelstm/preprocess_lcquad.py	——— preprocess LC-QUAD dataset
4. python learning/treelstm/main.py	——— train Tree-LSTM model
5. (Optional, the output already provided) python entity_lcquad_test.py	———  get the linked entities/relations for each question in LC-QUAD
6. python lcquad_test.py	——— get the generated SPARQL query for each question in LC-QUAD
7. python lcquad_result.py	——— analyze the generated SPARQL queries in LC-QUAD
8. (Optional, the output already provided) python qald_answer.py	———  get standard answers for each question in QALD7
9. (Optional, the output already provided) python entity_qald.py	———  get the linked entities/relations for each question in QALD7
10. python lcquad_test.py	——— get the generated SPARQL query for each question in QALD7
11. python qald_test.py	——— analyze the generated SPARQL queries in QALD7