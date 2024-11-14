def calculate_average_difference(file1, file2):
    try:
        # Open and read the content of both files
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
        
        # Check if both files have the same number of lines
        if len(lines1) != len(lines2):
            print("Error: The files do not have the same number of lines.")
            return
        
        # Calculate the differences and sum them
        total_difference = 0.0
        for line1, line2 in zip(lines1, lines2):
            try:
                num1 = float(line1.strip())
                num2 = float(line2.strip())
                total_difference += abs(num1 - num2)
            except ValueError:
                print("Error: Non-numeric data found in the files.")
                return
        
        # Calculate and print the average difference
        average_difference = total_difference / len(lines1)
        print(f"Average Difference: {average_difference}")

    except FileNotFoundError as e:
        print(f"Error: {e}")

# Example usage
file1 = 'my.ans'
file2 = 'cant/ans.mtx'
calculate_average_difference(file1, file2)

