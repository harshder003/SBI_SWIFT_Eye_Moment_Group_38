import pandas as pd
import matplotlib.pyplot as plt
import os

corpus_path = r"d:\MS_DS\SBI\Rcorpus_PB2.dat"
fixation_path = r"d:\MS_DS\SBI\fixseqin_PB2expVP10.dat"
output_dir = r"d:\MS_DS\SBI"

# Read Corpus Data
# It has a header, tab-separated
corpus_df = pd.read_csv(corpus_path, sep='\t')

# Read Fixation Data
# It doesn't seem to have a header, space/tab-separated
fixation_df = pd.read_csv(fixation_path, sep=r'\s+', header=None)
# Adding generic column names for fixation data based on typical eye-tracking datasets
fixation_df.columns = [f"col_{i}" for i in range(fixation_df.shape[1])]

# Calculate Summary Statistics
with open(os.path.join(output_dir, "summary_statistics.txt"), "w") as f:
    f.write("=== Corpus Data Summary ===\n")
    f.write(corpus_df.describe().to_string())
    f.write("\n\n=== Fixation Data Summary ===\n")
    f.write(fixation_df.describe().to_string())
    
print("Summary statistics saved to summary_statistics.txt")

# Create visualizations
# 1. Word Length Distribution (Corpus)
plt.figure(figsize=(8, 5))
corpus_df['length'].plot.hist(bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Word Lengths (Corpus)')
plt.xlabel('Word Length (characters)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'word_length_distribution.png'))
plt.close()

# 2. Word Frequency Distribution (Corpus) - using log scale as frequencies usually have a long tail
plt.figure(figsize=(8, 5))
corpus_df['freq'].plot.hist(bins=30, color='salmon', log=True, edgecolor='black')
plt.title('Distribution of Word Frequencies (Corpus, Log Scale)')
plt.xlabel('Word Frequency')
plt.ylabel('Count (Log Scale)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'word_freq_distribution.png'))
plt.close()

# 3. Fixation Duration Distribution (Assuming Column 3 is Duration based on typical values like 150-250ms)
plt.figure(figsize=(8, 5))
fixation_df['col_3'].plot.hist(bins=40, color='green', edgecolor='black')
plt.title('Distribution of Fixation Durations (Assuming Col 3)')
plt.xlabel('Fixation Duration (ms)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fixation_duration_distribution.png'))
plt.close()

# 4. Landing Position Distribution (Assuming Column 2 is character landing position)
plt.figure(figsize=(8, 5))
fixation_df['col_2'].plot.hist(bins=40, color='purple', edgecolor='black')
plt.title('Distribution of Landing Positions (Assuming Col 2)')
plt.xlabel('Landing Position (character index)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'landing_position_distribution.png'))
plt.close()

print("Visualizations saved as PNG files.")
