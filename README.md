# SemEval-2018-Task-1-Affect-in-Tweets
Affect in Tweets, which includes an array of subtasks on inferring the affectual state of a person from their tweet.
Background and Significance: We use language to communicate not only the emotion or sentiment we are feeling but also the intensity of the emotion or sentiment. For example, our utterances can convey that we are very angry, slightly sad, absolutely elated, etc. Here, intensity refers to the degree or amount of an emotion or degree of sentiment. We will refer to emotion-related categories such as anger, fear, sentiment, and arousal, by the term affect. Existing affect datasets are mainly annotated categorically without an indication of intensity. Further, past shared tasks have almost always been framed as classification tasks (identify one among n affect categories for this sentence). In contrast, it is often useful for applications to know the degree to which affect is expressed in text.

Tasks: We present an array of tasks where systems have to automatically determine the intensity of emotions (E) and intensity of sentiment (aka valence V) of the tweeters from their tweets. (The term tweeter refers to the person who has posted the tweet.) We also include a multi-label emotion classification task for tweets. For each task, we provide separate training and test datasets for English.The individual tasks are described below:

1. Task EI-oc: Detecting Emotion Intensity (ordinal classification)

Given:

    a tweet

    an emotion E (anger, fear, joy, or sadness)

Task: classify the tweet into one of four ordinal classes of intensity of E that best represents the mental state of the tweeter:

    0: no E can be inferred

    1: low amount of E can be inferred

    2: moderate amount of E can be inferred

    3: high amount of E can be inferred

For each language: 4 training sets and 4 test sets: one for each emotion E.

2. Task V-oc: Detecting Valence (ordinal classification) -- This is the traditional Sentiment Analysis Task

Given:

    a tweet

Task: classify the tweet into one of seven ordinal classes, corresponding to various levels of positive and negative sentiment intensity, that best represents the mental state of the tweeter:

    3: very positive mental state can be inferred

    2: moderately positive mental state can be inferred

    1: slightly positive mental state can be inferred

    0: neutral or mixed mental state can be inferred

    -1: slightly negative mental state can be inferred

    -2: moderately negative mental state can be inferred

    -3: very negative mental state can be inferred

For each language: 1 training set, 1 test set. 
