<Model><br>

1. Character Embedding Layer maps each word to a vector space using character-level
   CNNs.
   <s>2. Word Embedding Layer maps each word to a vector space using a pre-trained word embedding model.</s>
2. Contextual Embedding Layer utilizes contextual cues from surrounding words to refine
   the embedding of the words. These first three layers are applied to both the query and
   context.
3. Attention Flow Layer couples the query and context vectors and produces a set of queryaware feature vectors for each word in the context.
4. Modeling Layer employs a Recurrent Neural Network to scan the context.
5. Output Layer provides an answer to the query
   (7. Change for CNNDM)

Seungone : 1, 5, 6, 4(Q2C)
Hyungjoo : 2, 3, 4(C2Q) + Highway Network

<Data>
1. Huggingface datasets => SQuAD 1.0
2. Glove loader

<Experiments>
1. SQuAD 1.0 : Reading Comprehension(Extractive) / EM, F1 (Single Model, Ensemble)
2. CNN DailyDail : Reading Comprehension(Cloze Test) / Accuracy?

2. Ablation study:
   (1) No char emb
   (2) No word emb
   (3) No C2Q attn
   (4) No Q2C attn
   (5) Dynamic attn

3. Visualization
   (1) Visualize the feature spaces after the word and contextual embedding layers
   =>To visualize the embeddings, we choose a few frequent
   query words in the dev data(When, Where, Who, city, January, Seahawks, date) and look at the context words that have the highest cosine similarity to the query words
   => (GOAL) When begins to match years, Where matches locations, and Who matches
   names.

(2) visualize these two feature spaces using t-SNE +(Figure 2-b,c)

(3) visualize the attention matrices for some question-context tuples in the dev data

(4)
