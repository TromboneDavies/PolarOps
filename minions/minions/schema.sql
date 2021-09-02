
CREATE TABLE rated (comment_id text, rater text, rating text);

CREATE TABLE pile(
  "subreddit" TEXT,
  "submission_id" TEXT,
  "comment_id" TEXT,
  "text" TEXT,
  "date" TEXT,
  "batch_num" TEXT
);
