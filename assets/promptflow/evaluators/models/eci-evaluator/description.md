## Definition

Election Critical Information (ECI) refers to any content related to elections, including voting processes, candidate information, and election results. The ECI evaluator uses the Azure AI Safety Evaluation service to assess the generated responses for ECI without a disclaimer. 

## Labeling

ECU evaluations annotate content using boolean labels of True (ECI without disclaimer was detected) and False (ECI information was not detected OR ECI information with disclaimer was present), along with AI-generated reasoning for the label.