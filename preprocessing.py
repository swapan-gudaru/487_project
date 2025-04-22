import pandas as pd

sarcasm_df = pd.read_csv('sarcasm.csv')
sarcasm_df['Label'] = sarcasm_df['class'].map({'notsarc': 0, 'sarc': 1})
sarcasm_processed = sarcasm_df[['Label', 'text']].rename(columns={'text': 'Content'})

fb_comments = pd.read_csv('facebook_comments.csv')
fb_labels = pd.read_csv('facebook_labels.csv')
fb_processed = pd.DataFrame({
    'Label': fb_labels['Comments'],
    'Content': fb_comments['Comments']
})

twitter_comments = pd.read_csv('twitter_comments.csv')
twitter_labels = pd.read_csv('twitter_labels.csv')
twitter_processed = pd.DataFrame({
    'Label': twitter_labels['Comments'],
    'Content': twitter_comments['Comments']
})

reddit_df = pd.read_csv('reddit.csv')
reddit_processed = reddit_df[['label', 'comment']].rename(
    columns={'label': 'Label', 'comment': 'Content'})

combined_df = pd.concat([sarcasm_processed, fb_processed, twitter_processed, reddit_processed], 
                        ignore_index=True)
combined_df.insert(0, 'ID', range(1, len(combined_df) + 1))
combined_df.to_csv('combined_sarcasm_dataset.csv', index=False)
