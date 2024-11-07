import pandas as pd

def candidate_elimination(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Separate features and target
    features = data.columns[:-1]
    target = data.columns[-1]
    
    # Initialize S as the most specific hypothesis
    S = ['ϕ'] * len(features)
    # Initialize G as the most general hypothesis
    G = [['?'] * len(features)]
    
    # Print initial boundaries
    print("Initial Specific Boundary (S):", S)
    print("Initial General Boundary (G):", G)
    
    # Iterate through each training example
    for i, row in data.iterrows():
        print(f"\nProcessing example {i + 1}: {row.values}")
        
        if row[target] == 'Yes':  # Positive example
            # Update S
            for j in range(len(S)):
                if S[j] == 'ϕ':  # Initialize S with the first positive example
                    S[j] = row[features[j]]
                elif S[j] != row[features[j]]:
                    S[j] = '?'
            print("Updated Specific Boundary (S):", S)
            
            # Remove inconsistent hypotheses from G
            G = [g for g in G if all(g[k] == '?' or g[k] == row[features[k]] or S[k] == '?' for k in range(len(S)))]
            print("Filtered General Boundary (G) after positive example:", G)
        
        else:  # Negative example
            # Update G
            new_G = []
            for g in G:
                for j in range(len(g)):
                    if g[j] == '?' and S[j] != row[features[j]]:
                        # Create a more specific hypothesis and add it to G
                        new_hypothesis = g.copy()
                        new_hypothesis[j] = S[j]
                        if all(new_hypothesis[k] == row[features[k]] or new_hypothesis[k] == '?' for k in range(len(S))):
                            new_G.append(new_hypothesis)
            G = new_G
            print("Updated General Boundary (G) after negative example:", G)
    
    # Display the final version space boundaries
    print("\nFinal Specific Boundary (S):", S)
    print("Final General Boundary (G):", G)

# Example usage with corrected file path
file_path = r"C:\Users\kadiy\Documents\ML Practical\training_data.csv"  # Use raw string
candidate_elimination(file_path)
