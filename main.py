import pandas as pd
from KNN import KNN_train
from Native_Bayes import Native_bayes_train
from Support_Vector_Machine import SVM_train
from Random_forest import Random_forest_train
from Decission_tree import Decission_tree_train

# Loading the dataframe
def load_dataframe():
    df = pd.read_csv("Heart_Attack.csv")

    return df 

def main():
    df = load_dataframe()

    # Selection menu
    print("Select machine learning model:")
    print("------------------------------")
    print("(0) Exit")
    print("(1) Decision Tree")
    print("(2) Random Forest")
    print("(3) K-Nearest Neighbour")
    print("(4) Support Vector Machine (SVM)")
    print("(5) Native Bayes")
    print("(6) All the above")
    print("\n")

    # List with the selections
    selections = [0, 1, 2, 3, 4, 5, 6]

    # List with the algorithms
    algorithms = {
        1: ("Decision Tree", Decission_tree_train),
        2: ("Random Forest", Random_forest_train),
        3: ("K-Nearest Neighbour", KNN_train),
        4: ("Support Vector Machine (SVM)", SVM_train),
        5: ("Native Bayes", Native_bayes_train)
    }

    # Checking if the user selection is invalid
    while True:
        try:
            user_selection = int(input("Select the number of the model: "))
            print("\n")

            if user_selection in selections:
                break
            else:
                print("Oops! That was not a valid number. Try again...")
                print("\n")
        except ValueError:
            print("Oops! That was not a valid input. Please enter a number.")
            print("\n")

    # If the user's selection equals zero, the program stops.
    if user_selection == 0:
        print("Thank you for using my program!")
        exit(0)

    # If the user's selection matches one of the predefined options in the list [1, 2, 3, 4, 5],
    # retrieve the corresponding algorithm name and function from the 'algorithms' dictionary.
    # Then, print the algorithm name followed by a line of dashes equal in length to the name.
    # Finally, execute the selected function, passing in the dataframe 'df'. 
    elif user_selection in [1, 2, 3, 4, 5]:
        name, func = algorithms[user_selection]
        print(f"{name}:")
        print("-" * len(name))
        func(df)

    # If the user's selection equals to 6 the program is running all the algorithms
    elif user_selection == 6:
        for key, (name, func) in algorithms.items():
            print(f"{name}:")
            print("-" * (len(name)+1))
            func(df)
            print("\n")

if __name__ == "__main__":
    main()