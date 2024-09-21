from wordfreq import zipf_frequency

# Function to find and write words with Zipf frequency above and below a threshold to separate files
def filter_words_by_zipf(input_file, output_file_above, output_file_below, thres):
    # Open and read the input file
    with open(input_file, 'r') as file:
        words = file.read().split()  # Split by whitespace to get words

    # Filter words with Zipf frequency above and below the threshold
    words_above = [word for word in words if zipf_frequency(word, 'en') > thres]
    words_below = [word for word in words if zipf_frequency(word, 'en') <= thres]

    # Write the words above the threshold to the output file
    with open(output_file_above, 'w') as file:
        file.write(f"Word count: {len(words_above)}\n")
        words_above = list(set(words_above))
        for word in words_above:
            file.write(f"{word}\n")

    # Write the words below the threshold to the output file
    with open(output_file_below, 'w') as file:
        file.write(f"Word count: {len(words_below)}\n")
        words_below = list(set(words_below))
        for word in words_below:
            file.write(f"{word}\n")

# Example usage
input_file = '/home/dataset/Data/gw_data/download/stimuli/text/lw1.txt'  # Path to the input text file
output_file_above = 'common_words.txt'  # Path to the output file for words above the threshold
output_file_below = 'not_common_words.txt'  # Path to the output file for words below the threshold
filter_words_by_zipf(input_file, output_file_above, output_file_below, 4.0)

print(f"Words with Zipf frequency above and below the threshold have been written to {output_file_above} and {output_file_below}")