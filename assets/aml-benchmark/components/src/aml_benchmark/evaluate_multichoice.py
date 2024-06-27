import random

import numpy as np

# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.

    # print("a3", candidates)

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        # pred_index = random.choice(all_choices)
        pred_index = random.choice(list(index2ans.keys()))
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


if __name__ == "__main__":
    # all_choices = ["A", "B", "C", "D"]
    all_choices = []

    cases = [
        dict(
            response="To find the missing amounts for Company B, we can use the provided information. The formula for Net Income or (Loss) is:\n\n\\[ \\text{Net Income (or Loss)} = \\text{Revenues} + \\text{Gains} - \\text{Expenses} - \\text{Losses} \\]\n\nFor Company B, let's identify the known values and plug them into the formula:\n\n- Revenues = $1,480,500\n- Expenses = $1,518,300\n- Gains = ?\n- Losses = $0\n- Net Income = $39,690\n\nSubstituting the known values into the formula, we have:\n\n\\[ 39,690 = 1,480,500 + \\text{Gains} - 1,518,300 - 0 \\]\n\nRearranging to solve for Gains:\n\n\\[ 39,690 = 1,480,500 - 1,518,300 + \\text{Gains} \\]\n\n\\[ 39,690 = -37,800 + \\text{Gains} \\]\n\n\\[ \\text{Gains} = 39,690 + 37,800 = 77,490 \\]\n\nSo, the missing amount for Gains for Company B is $77,490. \n\nWe can summarize the missing amounts for Company B as follows:\n- Gains: $77,490",
            choices=['$63,020', '$58,410', '$71,320', '$77,490'],
            correct_choice="D",
        ),
        dict(
            response="To determine the ending balance in the owner's capital account, we need to calculate the net income or loss for December and adjust the beginning capital balance accordingly.\n\n1. Calculate Net Income (or Loss):\n   \\[\n   \\text{Net Income (or Loss)} = \\text{Revenue} - \\text{Expenses}\n   \\]\n   Revenue = \\$56,400\n   Expenses = \\$59,800\n\n   \\[\n   \\text{Net Income (or Loss)} = \\$56,400 - \\$59,800 = -\\$3,400\n   \\]\n\n   There was a net loss of \\$3,400 in December.\n\n2. Adjust the Beginning Capital Balance:\n   \\[\n   \\text{Ending Capital} = \\text{Beginning Capital} + \\text{Net Income (or Loss)}\n   \\]\n   Beginning Capital (December 1) = \\$175,300\n   Net Loss = -\\$3,400\n\n   \\[\n   \\text{Ending Capital} = \\$175,300 - \\$3,400 = \\$171,900\n   \\]\n\nTherefore, the ending balance in the owner's capital account is \\$171,900.",
            choices=['$171,900', '$170,000', '$172,500', '$181,900'],
            correct_choice="A",
        ),
        dict(
            response="To calculate the cash flow from assets, cash flow to creditors, and cash flow to stockholders for 2015, we need to first understand the formulas for these calculations.\n\n### 1. Cash Flow from Assets\nCash Flow from Assets = Operating Cash Flow - Net Capital Spending - Changes in Net Working Capital\n\n - **Operating Cash Flow (OCF)**\n\\[ OCF = EBIT + Depreciation - Taxes \\]\n\nWhere,\n\\[ EBIT = Sales - Cost of Goods Sold - Other Expenses - Depreciation \\]\n\nFor 2015:\n\\[ EBIT = \\$10,091 - \\$3,672 - \\$641 - \\$1,351 = \\$4,427 \\]\n\n\\[ OCF = \\$4,427 + \\$1,351 - \\$0 = \\$5,778 \\]\n(Assuming taxes are zero for simplicity)\n\n - **Net Capital Spending (NCS)**\n\\[ NCS = (Net Fixed Assets_{2015} - Net Fixed Assets_{2014}) + Depreciation \\]\n\n\\[ NCS = (42,332 - 41,346) + 1,351 = \\$1,985 \\]\n\n - **Changes in Net Working Capital (NWC)**\n\\[ NWC = (Current Assets_{2015} - Current Liabilities_{2015}) - (Current Assets_{2014} - Current Liabilities_{2014}) \\]\n\nFor simplicity, we will only consider Accounts Receivable, Inventory (Current Assets) and Short-term Notes Payable, Accounts Payable (Current Liabilities):\n\n\\[ Current Assets_{2015} = 6,244 (Cash) + 7,352 (Accounts Receivable) + 11,926 (Inventory) = 25,522 \\]\n\\[ Current Assets_{2014} = 4,931 (Cash) + 6,527 (Accounts Receivable) + 11,604 (Inventory) = 23,062 \\]\n\n\\[ Current Liabilities_{2015} = 895 (Short-term Notes Payable) + 5,022 (Accounts Payable) = 5,917 \\]\n\\[ Current Liabilities_{2014} = 953 (Short-term Notes Payable) + 5,179 (Accounts Payable) = 6,132 \\]\n\n\\[ \\Delta NWC = (25,522 - 5,917) - (23,062 - 6,132) = 19,605 - 16,930 = \\$2,675 \\]\n\n\\[ NWC = \\$2,675 \\]\n\nThus,\n\\[ Cash Flow from Assets = 5,778 - 1,985 - 2,675 = \\$1,118 \\]\n\n### 2. Cash Flow to Creditors\nCash Flow to Creditors = Interest Paid - Net New Borrowing\n\n\\[ Net New Borrowing = Long-term Debt_{2015} - Long-term Debt_{2014} \\]\n\n\\[ Net New Borrowing = 19,260 - 16,152 = 3,108 \\]\n\n\\[ Cash Flow to Creditors = 724 - 3,108 = - \\$2,384 \\]\n\n### 3. Cash Flow to Stockholders\nCash Flow to Stockholders = Dividends Paid - Net New Equity Raised\n\n(Note: No equity information is provided, so we assume it\u2019s zero for simplicity)\n\n\\[ Dividends Paid = 1,261 \\]\n\n\\[ Net New Equity Raised = 0 \\]\n\n\\[ Cash Flow to Stockholders = 1,261 \\]\n\n### Summary\n1. Cash Flow from Assets = \\$1,118\n2. Cash Flow to Creditors = - \\$2,384\n3. Cash Flow to Stockholders = \\$1,261",
            choices=['1): -$493.02 (2):-$2,384 (3):$1,890.98', '1): $1843.98 (2): -$2,384 (3):$493.02', '1): -$493.02 (2): -$2,384 (3):-$1,890.98'],
            correct_choice="C",
        ),
        dict(
            response="To determine the net cash flow per check from adopting the lockbox system, we need to calculate the benefits from reduced collection time and compare it to the cost of the lockbox service.\n\n### Step-by-step Calculation:\n\n1. **Calculate the daily interest savings per check due to the reduced collection time:**\n\n- \\( \\text{Average value of payment} = \\$865 \\)\n- \\( \\text{Daily interest rate} = 0.02\\% = 0.0002 \\)\n- \\( \\text{Time reduction in days} = 3 \\)\n\n\\[ \\text{Daily interest savings per check} = \\text{Average value of payment} \\times \\text{Daily interest rate} \\]\n\\[ \\text{Daily interest savings per check} = 865 \\times 0.0002 = \\$0.173 \\]\n\n\\[ \\text{Total interest savings for 3 days per check} = 0.173 \\times 3 = \\$0.519 \\]\n\n2. **Calculate the cost per check for using the lockbox service:**\n\n- \\( \\text{Variable lockbox fee (per transaction)} = \\$0.50 \\)\n\n3. **Determine the net cash flow per check:**\n\n\\[ \\text{Net cash flow per check} = \\text{Total interest savings for 3 days} - \\text{Variable lockbox fee} \\]\n\\[ \\text{Net cash flow per check} = 0.519 - 0.50 = \\$0.019 \\]\n\n### Conclusion:\n\nThe net cash flow per check from adopting the lockbox system is \\( \\$0.019 \\).",
            choices=['$.02', '$7.79', '$8.65'],
            correct_choice="A",
        ),
        dict(
            response="To solve for the unknown number of years, we can use the future value formula for compound interest:\n\n\\[ FV = PV \\times (1 + r)^n \\]\n\nWhere:\n- \\( FV \\) is the Future Value\n- \\( PV \\) is the Present Value\n- \\( r \\) is the annual interest rate (decimal)\n- \\( n \\) is the number of years\n\nWe need to solve for \\( n \\), so we rearrange the formula:\n\n\\[ n = \\frac{\\log(FV\/PV)}{\\log(1 + r)} \\]\n\nLet's compute the values for each row.\n\n**1.** Present Value = \\$625, Interest Rate = 7% (or 0.07), Future Value = \\$1,284\n\\[ n = \\frac{\\log(1284 \/ 625)}{\\log(1 + 0.07)} \\]\n\\[ n = \\frac{\\log(2.0544)}{\\log(1.07)} \\]\n\\[ n = \\frac{0.3120}{0.0294} \\]\n\\[ n \\approx 10.61 \\]\n\n**2.** Present Value = \\$810, Interest Rate = 12% (or 0.12), Future Value = \\$4,341\n\\[ n = \\frac{\\log(4341 \/ 810)}{\\log(1 + 0.12)} \\]\n\\[ n = \\frac{\\log(5.3617)}{\\log(1.12)} \\]\n\\[ n = \\frac{0.7291}{0.0471} \\]\n\\[ n \\approx 15.48 \\]\n\n**3.** Present Value = \\$16,500, Interest Rate = 17% (or 0.17), Future Value = \\$402,662\n\\[ n = \\frac{\\log(402662 \/ 16500)}{\\log(1 + 0.17)} \\]\n\\[ n = \\frac{\\log(24.4038)}{\\log(1.17)} \\]\n\\[ n = \\frac{1.3874}{0.0706} \\]\n\\[ n \\approx 19.64 \\]\n\n**4.** Present Value = \\$21,500, Interest Rate = 8% (or 0.08), Future Value = \\$147,350\n\\[ n = \\frac{\\log(147350 \/ 21500)}{\\log(1 + 0.08)} \\]\n\\[ n = \\frac{\\log(6.8547)}{\\log(1.08)} \\]\n\\[ n = \\frac{0.8358}{0.0334} \\]\n\\[ n \\approx 25.02 \\]\n\nSummarizing:\n1. Approximately 10.61 years\n2. Approximately 15.48 years\n3. Approximately 19.64 years\n4. Approximately 25.02 years",
            choices=['10.52 years; 14.73 years; 20.02 years; 24.73 years', '10.64 years; 14.81 years; 20.35 years; 25.01 years', '10.96 years; 15.22 years; 20.83 years; 25.96 years'],
            correct_choice="B",
        ),
    ]

    for i, c in enumerate(cases):
        response = c["response"]
        index2ans = {
            k: v
            for k, v in zip(["A", "B", "C", "D"], c["choices"])
        }
        correct_choice = c["correct_choice"]

        choice = parse_multi_choice_response(response, all_choices, index2ans)
        if choice != correct_choice:
            print(f"case {i} incorrect; choice {choice} correct choice {correct_choice}")
        else:
            print(f"case {i} correct;")
