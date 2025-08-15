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

def afficher_menu():
    print("\n=== MENU D'EXPÉRIMENTATION ===")
    for key, value in options.items():
        print(f"{key}. {value['name']}")
    print("0. Quitter")

def main():
    while True:
        afficher_menu()
        choix = input("\nSélectionnez une option: ").strip()
        
        if choix == "0":
            print("Au revoir !")
            break
        elif choix in options:
            cmd = options[choix]["cmd"]
            print("\n=== Ligne de commande actuelle ===")
            
            # Ligne pré-remplie éditable
            nouvelle_cmd = prompt(
                "\nModifiez la commande si nécessaire puis appuyez sur Entrée:\n> ",
                default=cmd
            ).strip()
            
            if nouvelle_cmd:
                cmd = nouvelle_cmd
            
            print(f"\n=== Exécution: {cmd} ===\n")
            os.system(cmd)
        else:
            print("Option invalide, veuillez réessayer.")

if __name__ == "__main__":
    main()
