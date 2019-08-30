# QAsparql
Question-Answering system for Knowledge Graph (DBpedia)

Preprocess:
1. bash earning/treelstm/download.sh -- download the pre-trained word embedding models FastText and Glove


Whole Process:
1. python lcquad_dataset.py -- preprocess the LC-QuAD dataset, generate 'linked_answer.json' file as the LC-QuAD dataset with golden standard answers
2. python lcquad_answer.py -- generate the golden answers for LC-QuAD dataset, generate 'lcquad_gold.json' file as LC-QuAD dataset with generated SPARQL queries based on the entities and properties extracted from the correct standard SPARQL query
3. python learning/treelstm/preprocess_lcquad.py -- preprocess the LC-QuAD dataset for Tree-LSTM training, split the original Lc-QuAD dataset into 'LCQuad_train.json', 'LCQuad_trial.json', 'LCQuad_test.json' each with 70%\20%\10% of the original dataset. Generate the dependency parsing tree and the corresponding input and output required to train the Tree-LSTM model.
4. python learning/treelstm/main.py -- train Tree-LSTM. The generated checkpoints files are stored in \checkpoints folder and used in lcquad_test.py and qald_test.py
5. python entity_lcquad_test.py -- generate phrase mapping for LC-QuAD test dataset
6. python entity_qald.py -- generate phrase mapping for QALD-7 test dataset
7. python lcquad_test.py -- test the QA system on LC-QuAD test dataset
8. python lcquadall_test.py -- test the QA system on LC-QuAD whole dataset
8. python qald_test.py  -- test the QA system on QALD-7 dataset
9. python question_type_anlaysis.py  -- analyze the question type classification accuracy on LC-QuAD and QALD-7 dataset
10. python result_analysis.py  -- analyze the final result for LC-QuAD and QALD-7 dataset


