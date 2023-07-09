import numpy as np
import matplotlib.pyplot as plt

colors = ['skyblue', 'lightgreen', 'lightcoral']
keywords = ['polusi udara', 'perubahan iklim', 'pemanasan global']
fig, ax = plt.subplots(figsize=(8, 7))
artists = []

for i, keyword in enumerate(keywords):
    data = tweet_tlwca[tweet_tlwca['keyword'] == keyword]['tweet_word_count']
    artist = ax.boxplot(data, positions=[i+1], widths=0.6, patch_artist=True)
    artists.append(artist['boxes'][0])

ax.set_xticks(range(1, len(keywords) + 1))
ax.set_xticklabels(keywords)
ax.set_xlim(0.5, len(keywords) + 0.5)
ax.set_ylabel('Tweet Word Count')

for artist, color in zip(artists, colors):
    artist.set(facecolor=color)

global_mean = np.mean(tweet_tlwca['tweet_word_count'])
ax.axhline(global_mean, linestyle='dashed', color='black')
mean_label = f'Mean: {global_mean:.2f}'
ax.text(len(keywords) + 0.6, global_mean, mean_label, ha='center', va='center',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
        color='black')

fig.suptitle('Boxplots of tweet word count', fontsize=16, fontweight='bold')
plt.show()
