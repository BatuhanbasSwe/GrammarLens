# Error Analysis: DeBERTa-v3 (seed 42)

This analysis looks at the test-set mistakes of DeBERTa-v3 (`results/preds/deberta-v3_seed42.csv`, 2,114 test examples, 101 wrong). It covers **seed 42 only**; the predictions of the other seeds are in `results/preds/` and were not reviewed here.

**Scope and method.** We reviewed the three largest confusion pairs, 51 of the 101 errors: `PREP → CORRECT` (24), `DET → CORRECT` (13) and `CORRECT → DET` (14). The other 50 errors were not reviewed. For each sentence we compared the model input with the error-free `source` sentence and judged whether the label is right. The judgments were made by **one annotator with AI assistance**, are subjective, and have no second opinion. Borderline cases are marked as such. Appendix A lists 20 representative sentences.

## Confusion matrix (rows: true label, columns: prediction)

| true \ pred | CORRECT | SVA | VERB_FORM | DET | NOUN_NUM | PREP | WORD_ORDER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `CORRECT` | 270 | 5 | 2 | 14 | 0 | 7 | 4 |
| `SVA` | 3 | 297 | 0 | 0 | 0 | 1 | 1 |
| `VERB_FORM` | 1 | 0 | 300 | 1 | 0 | 0 | 0 |
| `DET` | 13 | 1 | 4 | 281 | 0 | 2 | 1 |
| `NOUN_NUM` | 0 | 0 | 0 | 1 | 298 | 1 | 2 |
| `PREP` | 24 | 2 | 0 | 5 | 1 | 269 | 1 |
| `WORD_ORDER` | 1 | 0 | 1 | 1 | 0 | 1 | 298 |

## Findings

**1. The errors concentrate around `CORRECT`, `PREP` and `DET`.** 74 of the 101 errors involve `CORRECT`, and four confusions alone (`PREP→CORRECT` 24, `CORRECT→DET` 14, `DET→CORRECT` 13, `CORRECT→PREP` 7) make up 58 (57%). The other four classes are almost solved: recall is 98.3% for `SVA`, 99.3% for `VERB_FORM`, 98.7% for `NOUN_NUM` and 98.7% for `WORD_ORDER` (298 of 302 each, except `SVA` 297 and `VERB_FORM` 300). `WORD_ORDER` is solved even though a bag-of-words model cannot see it at all.

**2. Most `PREP → CORRECT` "errors" are label noise, not model errors.** Of the 24 sentences, 13 are still grammatical after the preposition swap (for example "a special connection **with** some extremists", "Starting **from** February", "dying **of** malaria"). 3 more are probably acceptable, 5 are borderline, 1 is a Usenet header ("Followup-With:") and not natural language. Only **2 are real misses**: "a dissenting opinion, **on** which THOMAS and ALITO joined" and "Are you free **from** lunch". The model answers `CORRECT` when the sentence sounds natural, which is the right behavior here.

**3. Most `DET → CORRECT` "errors" are also label noise.** Of the 13 sentences, 8 are still acceptable after the article was deleted or swapped (for example "loved by professional staff", "an example of "tragedy"", "adds tremendous value", "No good in bed"). 3 are borderline. 2 are real misses ("Girl raises her hand." and "…to see if product was in stock"). Deleting an article gives acceptable text before abstract or mass nouns and in headline or review style. In one case the a↔an swap even **repaired** an existing error: the original has "a excellent slide show", the "erroneous" version says "an excellent slide show".

**4. `CORRECT → DET`: the model catches errors that are already in the original text.** Of the 14 sentences, in 5 the original text clearly contains an article error ("a ebook", "a excellent slide show", "a "Earth-Moon" highway", "a bronze one and marble one", "Here is latest draft"). In 5 more an article is missing in an informal style ("lecturing in USA", "Scheduled appointment for…", "plenty of things to do near hotel", "stronger will than you think", "Took 1+ hour to deliver…"). Only **4 are real false alarms**. The cleaning step only removes sentences that the treebank itself marks as typos (rule 1), not grammar mistakes in web text such as reviews, e-mails and forum posts. So `CORRECT` is not perfectly clean. The sentence "a excellent slide show" shows both effects at once: the original (wrong English) is labeled `CORRECT`, the repaired version is labeled `DET`, and the model predicts the opposite of both labels, which is the right answer for both.

**5. The measured score probably underestimates the model, but we cannot say by how much.** In the reviewed sample, 8 of 13 `DET→CORRECT` mistakes (11 counting the borderline ones) and 13 of 24 `PREP→CORRECT` mistakes (16 counting the three probably acceptable ones) look like label problems, and 10 of 14 `CORRECT→DET` mistakes come from errors in the original text. This suggests the reported macro-F1 is somewhat pessimistic. The sample is not random (it contains only sentences the model got wrong, so noisy examples are over-represented), therefore the share of noisy labels in the whole class cannot be estimated from it. It is a lower bound on the noise that the model exposed, not a rate. BLiMP results should be read next to this analysis, because BLiMP pairs are constructed by hand.

**6. Why it happens, and what we did not do.** The injected `PREP` and `DET` errors are single edits that often leave natural text (many prepositions are interchangeable, articles are optional in many contexts), while `SVA`, `VERB_FORM`, `NOUN_NUM` and `WORD_ORDER` edits almost always break the sentence. A possible fix is to filter out injected sentences that still look acceptable (for example with a language model score or a grammar checker) and to drop original sentences with article errors. **We did not do this**; it is a suggestion for future work and no result for it is reported here. The same limits are listed in `docs/preprocessing.md` (Sections 5 and 6).

## Appendix A: 20 reviewed sentences

Order numbers refer to the order of rows in the pair (rows of `deberta-v3_seed42.csv` filtered by true label and prediction, in file order, joined with `data/processed/test.csv` on `id`).

| Pair, # | Model input | Original | Our reading |
| --- | --- | --- | --- |
| `PREP→CORRECT` 7 | They also had a special connection **with** some extremists in Jordan and Germany. | …connection **to**… | Still correct, label noise |
| `PREP→CORRECT` 9 | **Starting from** February, you will be able to export the data… | **Starting in** February… | Still correct, label noise |
| `PREP→CORRECT` 12 | I just wanted to check **on** you regarding the consulting arrangement… | …check **with** you… | Still correct, label noise |
| `PREP→CORRECT` 18 | His artworks were selected and exhibited **at** British Museum in 2002. | …exhibited **in** British Museum… | Still correct (arguably better than the original), label noise |
| `PREP→CORRECT` 22 | Because the 10.000.000 people dying **of** malaria will otherwise be dead. | …dying **from** malaria… | Still correct, label noise |
| `PREP→CORRECT` 4 | SCALIA filed a dissenting opinion, **on** which THOMAS and ALITO joined. | …**in** which… | Real error, model missed it |
| `PREP→CORRECT` 15 | Are you free **from** lunch some day this week? | …free **for** lunch… | Real error, model missed it |
| `DET→CORRECT` 1 | Girl raises her hand. | **A** girl raises her hand. | Real error (headline style), model missed it |
| `DET→CORRECT` 2 | …"No good in bed, but fine against a wall." | …"No good in **a** bed…" | Still acceptable (idiom), label noise |
| `DET→CORRECT` 3 | John Donovan from Argghhh! has put out **an** excellent slide show… | …put out **a** excellent slide show… | The "error" is correct English, label noise |
| `DET→CORRECT` 5 | Your children will be taken care of and loved by professional staff. | …by **a** professional staff. | Still acceptable, label noise |
| `DET→CORRECT` 9 | …if anyone can give him an example of "tragedy". | …an example of **a** "tragedy". | Still acceptable, label noise |
| `DET→CORRECT` 11 | I don't want to have to deal with those deal--day websites like Groupon. | …deal-**a**-day websites… | Garbled, but the deleted "a" is inside a compound, borderline |
| `DET→CORRECT` 13 | Travelled 40mins after calling to see if product was in stock. | …if **a** product was in stock. | Real error (informal), model missed it |
| `CORRECT→DET` 3 | don't forget to use a calcium supplement twice a week; captive reptiles are prone to calcium deficiency. | (same) | Looks correct, real false alarm |
| `CORRECT→DET` 4 | …(it was actually a ebook)… | (same) | Original contains the article error, model is right |
| `CORRECT→DET` 9 | With their proposal of a "Earth-Moon" highway, it looks like space tourism may become a reality… | (same) | Original contains the article error, model is right |
| `CORRECT→DET` 10 | Here is latest draft of risk memo (STILL IN DRAFT FORM). | (same) | Original lacks "the", model is right |
| `CORRECT→DET` 12 | Good local steakhouse, I recommend it! | (same) | Headline-style fragment, real false alarm |
| `CORRECT→DET` 14 | John Donovan from Argghhh! has put out a excellent slide show on what was actually found and fought for in Fallujah | (same) | Original contains the article error, model is right |
