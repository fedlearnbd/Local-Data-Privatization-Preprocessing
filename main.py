import os
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import CompleteStyle

options = {

    "1": {"name": "Analysis of distances between original and private data (LDPP) - (Distance Calculation)", 
          "cmd": "python experiments/run_ldpp_distance_analysis.py --dataset pathmnist --epsilon 3.0 --noise gaussian --transform normal --samples 1000"},

    "2": {"name": "Analysis of distances between original and private data (Naive DP) - (Distance Calculation)", 
          "cmd": "python experiments/run_direct_noise_injection_distance_analysis.py --dataset pathmnist --epsilon 3.0 --noise gaussian --samples 1000"},

    "3": {"name": "Federated Averaging - (FedAvg)", "cmd": "python experiments/run_fedavg.py --dataset pathmnist --rounds 10"},
    
    "4": {"name": "Federated With Krum Aggregation - (FedKrum)", "cmd": "python experiments/run_fedkrum.py --dataset pathmnist --rounds 10"},

    "5": {"name": "Ldpp with Federated Averaging - (Ldpp-avg)", "cmd": "python experiments/run_ldpp_avg.py --dataset pathmnist --rounds 10 --transform normal --noise gaussian --epsilon 3.0"},

    "6": {"name": "Ldpp with Krum Aggregation - (Ldpp-krum)", "cmd": "python experiments/run_ldpp_krum.py --dataset pathmnist --rounds 10 --transform normal --noise gaussian --epsilon 3.0"},

    "7": {"name": "Membership Inference Attack For Fedavg - (MIA-Fedavg)", "cmd": "python experiments/run_for_mia_fedavg.py --dataset pathmnist --rounds 10 --samples 1000"},

    "8": {"name": "Membership Inference Attack For Ldpp-Avg - (MIA-Ldpp-Avg)", 
          "cmd": "python experiments/run_for_mia_ldpp_avg.py --dataset pathmnist --rounds 10 --samples 1000 --transform normal --noise gaussian --epsilon 3.0"},

    "9": {"name": "Membership Inference Attack For Ldpp-Krum - (MIA-Ldpp-Krum)", 
          "cmd": "python experiments/run_for_mia_ldpp_krum.py --dataset pathmnist --rounds 10 --samples 1000 --transform normal --noise gaussian --epsilon 3.0"},

    "10": {"name": "Label Flipping Attack For Federated Averaging - (LFA-FedAvg)", 
           "cmd": "python experiments/run_for_lfa_fedavg.py --dataset pathmnist --rounds 10 --attacker_ratio 0.3"},
    
    "11": {"name": "Label Flipping Attack For Ldpp-Avg - (LFA-Ldpp-Avg)", 
           "cmd": "python experiments/run_for_lfa_ldpp_avg.py --dataset pathmnist --rounds 10  --transform normal --noise gaussian --epsilon 3.0 --attacker_ratio 0.3"},
    
    "12": {"name": "Label Flipping Attack For Ldpp-Krum - (LFA-Ldpp-Avg)", 
           "cmd": "python experiments/run_for_lfa_ldpp_krum.py --dataset pathmnist --rounds 10 --transform normal --noise gaussian --epsilon 3.0 --attacker_ratio 0.3"},

}

def show_menu():
    print("\n=== EXPERIMENTATION MENU ===")
    for key, value in options.items():
        print(f"{key}. {value['name']}")
    print("0. Exit")

def main():
    while True:
        show_menu()
        choice = input("\nSelect an option: ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice in options:
            cmd = options[choice]["cmd"]
            print("\n=== Current Command ===")
            
            # Editable pre-filled command line
            new_cmd = prompt(
                "\nEdit the command if necessary and press Enter:\n> ",
                default=cmd
            ).strip()
            
            if new_cmd:
                cmd = new_cmd
            
            print(f"\n=== Executing: {cmd} ===\n")
            os.system(cmd)
        else:
            print("Invalid option, please try again.")

if __name__ == "__main__":
    main()
