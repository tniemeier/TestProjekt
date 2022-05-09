import matplotlib.pyplot as plt
import numpy as np

skill = {'stringing': [0.36666667461395264, 0.3761904835700989], 'completion': [0.9047619104385376, 0.9142857193946838], 'under_extrusion': [0.961904764175415, 0.961904764175415], 'warping': [0.4476190507411957, 0.43809524178504944], 'lost_adhesion': [0.8142856955528259, 0.8142856955528259], 'gaps': [0.9285714030265808, 0.9238095283508301], 'over_extrusion': [0.9952380657196045, 0.9952380657196045], 'not_labelable': [1.0, 1.0], 'layer_separation': [0.9523809552192688, 0.9285714030265808], 'line_misalignment': [0.6047618985176086, 0.6000000238418579], 'layer_misalignment': [0.723809540271759, 0.6857143044471741], 'burning': [0.9571428298950195, 0.9571428298950195], 'overall_ok': [0.7476190328598022, 0.7714285850524902], 'blobs': [0.8523809313774109, 0.8523809313774109], 'poor_bridging': [0.5761904716491699, 0.776190459728241]}
skill_std = {}
skill_mean = {}

for key in skill:
    skill_std[key] = np.std(skill[key])
    skill_mean[key] = np.mean(skill[key])

labels = list(skill.keys())
val_std = list(skill_std.values())
val_mean  = list(skill_mean.values())

x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, val_std, width, label='std')

fig2, ax2 = plt.subplots()
rects2 = ax2.bar(x, val_mean, width, label='mean')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by attributes in std')
ax.set_xticks(x)
ax.tick_params(labelrotation=90)
ax.set_xticklabels(labels)

ax2.set_ylabel('Scores')
ax2.set_title('Scores by attributes in mean')
ax2.set_xticks(x)
ax2.tick_params(labelrotation=90)
ax2.set_xticklabels(labels)

fig.tight_layout()
fig2.tight_layout()
plt.show()