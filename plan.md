## Model
<br>
1. <s>Character Embedding Layer maps each word to a vector space using character-level CNNs.</s> <br>
2. <s>Word Embedding Layer maps each word to a vector space using a pre-trained word embedding model.</s> <br>
3. Contextual Embedding Layer utilizes contextual cues from surrounding words to refinethe embedding of the words. <br>
   These first three layers are applied to both the query and context. <br>
4. Attention Flow Layer couples the query and context vectors and produces a set of queryaware feature vectors for each word in the context. <br>
5. Modeling Layer employs a Recurrent Neural Network to scan the context. <br>
6. Output Layer provides an answer to the query <br>
7. Apply Changes for CNNDM <br>
<br>
Seungone : 1, 5, 6, 4(Q2C) <br>
Hyungjoo : 2, 3, 4(C2Q) + Highway Network <br>
<br>

## Data
<br>
1. <s>Huggingface datasets => SQuAD 1.0</s> <br>
2. <s>Glove loader</s> <br>
<br>

## Experiments
1. SQuAD 1.0 : Reading Comprehension(Extractive) / EM, F1 (Single Model, Ensemble) <br>
2. CNN DailyDail : Reading Comprehension(Cloze Test) / Accuracy? <br>
<br>

## Ablation study <br>
   (1) No char emb <br>
   (2) No word emb <br>
   (3) No C2Q attn <br>
   (4) No Q2C attn <br>
   (5) Dynamic attn <br>
<br>

## Visualization <br>
(1) Visualize the feature spaces after the word and contextual embedding layers <br>
   => To visualize the embeddings, we choose a few frequent query words in the dev data(When, Where, Who, city, January, Seahawks, date) <br>
   => look at the context words that have the highest cosine similarity to the query words <br>
   => (GOAL) When begins to match years, Where matches locations, and Who matches names. <br>
<br>
(2) visualize these two feature spaces using t-SNE +(Figure 2-b,c) <br>
<br>
(3) visualize the attention matrices for some question-context tuples in the dev data <br>
<br>