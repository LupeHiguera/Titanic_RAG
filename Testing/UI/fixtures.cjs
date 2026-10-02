// Illustrative UI fixtures only. These passages are NOT historical quotations.
const witnesses = [
  "Charles Herbert Lightoller",
  "Frederick Fleet",
  "J. Bruce Ismay",
];
const longPassage =
  "Illustrative test passage — not a historical quotation. Senator SMITH. What information was available to you that evening? Mr. ISMAY. The messages concerned ice along the route. The precise time and the sequence of events would need to be checked against the original record. Senator SMITH. Was that information brought to the attention of the officers? Mr. ISMAY. That is the detail on which I would ask you to consult the inquiry transcript.";
const results = [
  {
    witness_name: "J. Bruce Ismay",
    role: "Managing Director, White Star Line",
    source_type: "us_inquiry",
    page_number: 8,
    content: longPassage.replace("ice", "**ice**"),
    similarity_score: 0.724,
    relevance_score: 0.71,
    explanation: "Fixture explanation: this passage mentions the topic.",
  },
  {
    witness_name: "Charles Herbert Lightoller",
    role: "Second Officer, Titanic",
    source_type: "british_inquiry",
    page_number: 310,
    content:
      "Illustrative test passage — not a historical quotation. The account describes the conditions on watch and the handling of messages about ice. Read the cited transcript to examine the witness’s own words.",
    similarity_score: 0.681,
    relevance_score: 0.67,
  },
  {
    witness_name: "Frederick Fleet",
    role: "Lookout, Titanic",
    source_type: "british_inquiry",
    page_number: 420,
    content:
      "Illustrative test passage — not a historical quotation. This account concerns visibility from the lookout position and the observations made before the collision.",
    similarity_score: 0.644,
    relevance_score: 0.64,
  },
];
const contradictions = [
  {
    witness_a: "J. Bruce Ismay",
    witness_b: "J. Bruce Ismay",
    role_a: "Managing Director, White Star Line",
    role_b: "Managing Director, White Star Line",
    source_a: "us_inquiry",
    source_b: "british_inquiry",
    page_a: 8,
    page_b: 440,
    claim_a: "The message was received before the watch changed.",
    claim_b: "The message was received after the watch changed.",
    chunk_a: longPassage,
    chunk_b: longPassage,
    confidence: 0.85,
    same_person: true,
    explanation:
      "Illustrative model output: the two paraphrases place the message on different sides of the same event. This is a UI fixture, not a finding about this witness.",
  },
];
module.exports = { witnesses, results, contradictions };
