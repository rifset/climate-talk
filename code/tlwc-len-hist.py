import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
colors = ['skyblue', 'lightgreen', 'lightcoral']
keywords = ['polusi udara', 'perubahan iklim', 'pemanasan global']

for i, keyword in enumerate(keywords):
    data = tweet_tlwca[tweet_tlwca['keyword'] == keyword]['tweet_len']
    axes[i].hist(data, bins=20, color=colors[i])
    axes[i].set_title(keyword)

plt.suptitle('Tweet length distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
