"""Phase 2: Build V3 evaluation dataset with 120 grounded questions."""
import json
from pathlib import Path

def build_v3_test_set():
    """Build V3 evaluation dataset grounded in actual source documents."""
    
    questions = []
    qid = 0
    
    def add(cat, question, answerable, sources, sections, pages, concepts, difficulty, intent, notes):
        nonlocal qid
        qid += 1
        questions.append({
            "id": f"V3-{qid:03d}",
            "question": question,
            "category": cat,
            "answerable": answerable,
            "expected_sources": sources,
            "expected_sections": sections,
            "expected_pages": pages,
            "required_concepts": concepts,
            "difficulty": difficulty,
            "intent": intent,
            "notes": notes,
        })

    # ==================================================================
    # CATEGORY 1: Diabetes Diagnostic Criteria (Table 2.1, ADA p2)
    # ==================================================================
    add("diagnostic_criteria", "What A1C level is considered diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["A1C", "6.5%", "48 mmol/mol"], "easy", "diagnostic_criteria",
        "Table 2.1: A1C >=6.5% (>=48 mmol/mol)")
    add("diagnostic_criteria", "What is the fasting plasma glucose threshold for diagnosing diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["FPG", "126 mg/dL", "7.0 mmol/L"], "easy", "diagnostic_criteria",
        "Table 2.1: FPG >=126 mg/dL (>=7.0 mmol/L)")
    add("diagnostic_criteria", "What is the 2-hour OGTT threshold for diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["OGTT", "200 mg/dL", "11.1 mmol/L"], "easy", "diagnostic_criteria",
        "Table 2.1: 2-h PG >=200 mg/dL (>=11.1 mmol/L)")
    add("diagnostic_criteria", "What random plasma glucose level indicates diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["random PG", "200 mg/dL", "symptoms"], "easy", "diagnostic_criteria",
        "Table 2.1: random PG >=200 mg/dL with classic symptoms")
    add("diagnostic_criteria", "What tests can be used to diagnose diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["A1C", "FPG", "OGTT", "random PG"], "easy", "diagnostic_criteria",
        "Table 2.1 lists all four diagnostic tests")
    add("diagnostic_criteria", "Can CGM be used to diagnose diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["CGM", "insufficient evidence", "not recommended"], "medium", "diagnostic_criteria",
        "p2: insufficient evidence for CGM screening/diagnosis")

    # ==================================================================
    # CATEGORY 2: Prediabetes Criteria (ADA p2, p10)
    # ==================================================================
    add("prediabetes", "What are the diagnostic criteria for prediabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes"], [2, 10],
        ["IFG", "IGT", "A1C 5.7-6.4%"], "medium", "prediabetes_criteria",
        "Multiple criteria across p2 and p10")
    add("prediabetes", "What is the fasting glucose range for prediabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes"], [10],
        ["IFG", "100-125 mg/dL", "5.6-6.9 mmol/L"], "easy", "prediabetes_criteria",
        "p10: IFG defined as FPG 100-125 mg/dL (5.6-6.9 mmol/L)")
    add("prediabetes", "What A1C range indicates prediabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes"], [10],
        ["A1C", "5.7-6.4%", "39-47 mmol/mol"], "easy", "prediabetes_criteria",
        "p10: A1C 5.7-6.4% (39-47 mmol/mol)")
    add("prediabetes", "What is the OGTT threshold for prediabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes"], [10],
        ["IGT", "140-199 mg/dL", "7.8-11.0 mmol/L"], "easy", "prediabetes_criteria",
        "p10: IGT = 2-h PG 140-199 mg/dL (7.8-11.0 mmol/L)")
    add("prediabetes", "What is the difference between impaired fasting glucose and impaired glucose tolerance?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes"], [10],
        ["IFG", "IGT", "fasting vs 2-hour", "different tests"], "medium", "prediabetes_criteria",
        "IFG = fasting test 100-125; IGT = 2-hour test 140-199")

    # ==================================================================
    # CATEGORY 3: A1C Technical (ADA p2-4)
    # ==================================================================
    add("a1c", "What are the advantages of A1C over FPG and OGTT?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [3],
        ["convenience", "no fasting", "stability", "perturbations"], "medium", "a1c",
        "p3: convenience (no fasting), preanalytical stability, fewer day-to-day perturbations")
    add("a1c", "What conditions interfere with A1C testing?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [3, 4],
        ["hemoglobin variants", "G6PD", "pregnancy", "RBC turnover"], "medium", "test_interference",
        "p3-4: hemoglobin variants, G6PD deficiency, pregnancy, altered RBC turnover, etc.")
    add("a1c", "How should A1C be used for diagnosis in the US?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2, 3],
        ["NGSP", "DCCT", "certified lab", "POC restrictions"], "medium", "a1c",
        "p2-3: NGSP certified, standardized to DCCT, POC testing restrictions")
    add("a1c", "What are the A1C advantages over glucose-based tests?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [3],
        ["convenience", "no fasting", "stability", "perturbations"], "easy", "a1c",
        "p3: convenience, preanalytical stability, fewer day-to-day perturbations")
    add("a1c", "What affects A1C results according to the ADA standards?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [3, 4],
        ["hemoglobin variants", "erythropoietin", "iron deficiency", "kidney disease"], "hard", "test_interference",
        "p3-4: multiple interfering conditions listed in Table 2.3")
    add("a1c", "What is the recommended frequency for A1C testing in diabetes monitoring?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [3],
        ["monitoring", "quarterly", "every 3 months"], "medium", "a1c",
        "p3: A1C reflects 2-3 month glucose exposure; at least twice yearly")

    # ==================================================================
    # CATEGORY 4: FPG (ADA p2, NIDDK p1)
    # ==================================================================
    add("fpg", "What does fasting mean for the FPG test?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["no caloric intake", "8 hours"], "easy", "fpg",
        "p2 Table 2.1 footnote: no caloric intake for at least 8 h")
    add("fpg", "What is the coefficient of variation for the FPG test?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["5.7%", "biological variation", "110-142 mg/dL"], "hard", "fpg",
        "NIDDK: CV of 5.7%, result of 126 could indicate true FPG of 110-142")
    add("fpg", "What is the sensitivity of the FPG test compared to A1C?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["FPG sensitivity", "greater than A1C"], "medium", "test_comparison",
        "NIDDK: FPG sensitivity greater than A1C test")
    add("fpg", "What is sample stability for the FPG test?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["low stability", "30 minutes", "sodium fluoride"], "medium", "fpg",
        "NIDDK: low sample stability, requires processing within 30 minutes")

    # ==================================================================
    # CATEGORY 5: OGTT (ADA p2, NIDDK p1)
    # ==================================================================
    add("ogtt", "What is the 2-h glucose load for the OGTT?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["75 grams", "anhydrous glucose", "WHO protocol"], "easy", "ogtt",
        "p2 Table 2.1: 75 g anhydrous glucose dissolved in water per WHO")
    add("ogtt", "What patient preparation is required before an OGTT?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["carbohydrate intake", "150 g/day", "3 days", "8-hour fast"], "medium", "ogtt",
        "NIDDK: at least 150 g/day carbs for 3 days before test, 8-hour fast")
    add("ogtt", "How many samples are needed for an OGTT?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["two samples", "fasting", "2 hours"], "medium", "ogtt",
        "NIDDK: two samples after 8-hour fast and 2 hours after glucose load")

    # ==================================================================
    # CATEGORY 6: Random Plasma Glucose (ADA p2)
    # ==================================================================
    add("random_pg", "When is random plasma glucose used for diabetes diagnosis?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["symptoms", "hyperglycemia", "crisis", "polyuria", "polydipsia"], "easy", "random_glucose",
        "p2: only with classic symptoms of hyperglycemia or hyperglycemic crisis")

    # ==================================================================
    # CATEGORY 7: Confirmation of Diagnosis (ADA p4)
    # ==================================================================
    add("confirmation", "How should a diabetes diagnosis be confirmed?",
        True, ["ada_soc_2026_diagnosis"], ["Confirming the Diagnosis"], [4],
        ["two abnormal results", "different tests", "same test different time"], "hard", "diagnosis_confirmation",
        "p4: two abnormal results from different tests or same test at different time points")
    add("confirmation", "What confirmatory tests are recommended after an initial diabetes diagnosis?",
        True, ["ada_soc_2026_diagnosis"], ["Confirming the Diagnosis"], [4],
        ["repeat A1C", "OGTT", "confirmatory"], "hard", "diagnosis_confirmation",
        "p4: preferably repeat A1C or OGTT")
    add("confirmation", "Can a diabetes diagnosis be made with a single test?",
        True, ["ada_soc_2026_diagnosis"], ["Confirming the Diagnosis"], [4],
        ["except RPG", "hyperglycemic crisis", "confirm needed"], "medium", "diagnosis_confirmation",
        "p4: diagnosis requires confirmation except in symptomatic hyperglycemia")

    # ==================================================================
    # CATEGORY 8: Screening (ADA p11-12)
    # ==================================================================
    add("screening", "How often should people at risk be screened for diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults"], [11, 12],
        ["every 3 years", "risk factors"], "medium", "screening",
        "p11-12: screening every 3 years for adults at risk")
    add("screening", "What are the risk factors for type 2 diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults"], [11],
        ["age", "obesity", "inactivity", "prediabetes", "GDM", "PCOS", "hypertension"], "medium", "screening",
        "p11: age, obesity, physical inactivity, prediabetes, prior GDM, PCOS, hypertension")
    add("screening", "When should screening for type 2 diabetes begin in children?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults"], [10],
        ["after puberty", "age 10", "overweight", "risk factors"], "hard", "screening",
        "p10 Table 2.6: after puberty or age 10, with overweight/obesity and risk factors")

    # ==================================================================
    # CATEGORY 9: Type 1 Diabetes (ADA p6-8)
    # ==================================================================
    add("type1", "What are the stages of type 1 diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Type 1 Diabetes"], [6, 7],
        ["Stage 1", "Stage 2", "Stage 3", "autoimmunity", "dysglycemia"], "medium", "type1",
        "p6-7: Stage 1 autoimmunity normoglycemia, Stage 2 dysglycemia, Stage 3 clinical")
    add("type1", "What autoantibodies indicate type 1 diabetes risk?",
        True, ["ada_soc_2026_diagnosis"], ["Type 1 Diabetes"], [7],
        ["IAA", "GADA", "IA-2A", "ZnT8A", "ICA"], "hard", "type1",
        "p7: IAA, GADA, IA-2A, ZnT8A, ICA")
    add("type1", "How is risk stratified for progression to type 1 diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Type 1 Diabetes"], [7, 8],
        ["number of autoantibodies", "age", "metabolic staging"], "hard", "type1",
        "p7-8: risk increases with number of autoantibodies and metabolic changes")

    # ==================================================================
    # CATEGORY 10: Type 2 Diabetes (ADA p10-12)
    # ==================================================================
    add("type2", "What are the characteristics of type 2 diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes"], [10],
        ["insulin resistance", "beta cell dysfunction", "relative deficiency"], "medium", "type2",
        "p10: characterized by insulin resistance and relative insulin deficiency")
    add("type2", "What conditions are associated with increased type 2 diabetes risk?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults"], [11],
        ["obesity", "PCOS", "prediabetes", "hypertension", "dyslipidemia"], "medium", "screening",
        "p11: multiple risk factors including obesity, PCOS, hypertension")

    # ==================================================================
    # CATEGORY 11: Gestational Diabetes (ADA p16-17)
    # ==================================================================
    add("gestational", "How is gestational diabetes diagnosed?",
        True, ["ada_soc_2026_diagnosis"], ["Gestational Diabetes Mellitus"], [16, 17],
        ["75-g OGTT", "24-28 weeks", "IADPSG", "two-step"], "medium", "gestational",
        "p16-17: 75-g OGTT at 24-28 weeks, IADPSG thresholds or two-step approach")
    add("gestational", "What are the risk factors for gestational diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Gestational Diabetes Mellitus"], [16],
        ["overweight", "obesity", "family history", "prior GDM"], "medium", "gestational",
        "p16: references Table 2.5 for risk factors")
    add("gestational", "What glucose thresholds are used for GDM in the one-step approach?",
        True, ["ada_soc_2026_diagnosis"], ["Gestational Diabetes Mellitus"], [16, 17],
        ["fasting 92", "1-h 180", "2-h 153", "IADPSG"], "hard", "gestational",
        "p17: fasting >=92, 1-h >=180, 2-h >=153 mg/dL (IADPSG)")

    # ==================================================================
    # CATEGORY 12: Monogenic Diabetes (ADA p15-16)
    # ==================================================================
    add("monogenic", "What are the monogenic diabetes syndromes?",
        True, ["ada_soc_2026_diagnosis"], ["Monogenic Diabetes Syndromes"], [15, 16],
        ["MODY", "HNF1A", "GCK", "HNF4A", "neonatal"], "medium", "monogenic",
        "p15-16: MODY (HNF1A, GCK, HNF4A), neonatal diabetes")
    add("monogenic", "How is MODY different from type 1 and type 2 diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Monogenic Diabetes Syndromes"], [15, 16],
        ["monogenic", "autosomal dominant", "single gene", "incorrect diagnosis"], "hard", "monogenic",
        "p15: monogenic forms often misdiagnosed as type 1 or type 2")

    # ==================================================================
    # CATEGORY 13: Special Populations (ADA p13-14)
    # ==================================================================
    add("special_populations", "How is cystic fibrosis-related diabetes diagnosed?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes"], [13, 14],
        ["A1C >=6.5%", "OGTT", "within 3 months"], "hard", "special_population",
        "p13-14: A1C >=6.5% or OGTT within 3 months")
    add("special_populations", "What screening recommendations exist for diabetes in cystic fibrosis?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes"], [13, 14],
        ["annual OGTT", "CFRD"], "hard", "special_population",
        "p13-14: annual OGTT screening recommended")

    # ==================================================================
    # CATEGORY 14: A1C Interference (ADA p3-4)
    # ==================================================================
    add("a1c_interference", "Which hemoglobin variants affect A1C testing?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [3],
        ["HbC", "HbS", "HbE", "HbD", "African", "Mediterranean"], "hard", "test_interference",
        "p3: HbC, HbS, HbE, HbD traits; affects people of African, Mediterranean, SE Asian heritage")
    add("a1c_interference", "In which populations should alternative tests to A1C be considered?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [3, 4],
        ["sickle cell", "hemoglobin variants", "African American"], "hard", "test_interference",
        "p3-4: populations with hemoglobin variants, sickle cell trait")

    # ==================================================================
    # CATEGORY 15: Test Comparison (NIDDK p1)
    # ==================================================================
    add("test_comparison", "What are the pros and cons of the FPG test?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["low cost", "automated", "fasting required", "diurnal variation"], "medium", "test_comparison",
        "NIDDK: pros - low cost, automated; cons - fasting required, diurnal variation")
    add("test_comparison", "What are the pros and cons of the OGTT test?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["sensitive", "early marker", "not convenient", "expensive"], "medium", "test_comparison",
        "NIDDK: pros - sensitive, early marker; cons - inconvenient, expensive")
    add("test_comparison", "What are the pros and cons of the A1C test?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["no fasting", "standardization", "insensitive", "cost"], "medium", "test_comparison",
        "NIDDK: pros - no fasting, established standardization; cons - insensitive, higher cost")
    add("test_comparison", "What are the differences between FPG, A1C, and OGTT tests?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["comparison", "sensitivity", "convenience", "cost"], "hard", "test_comparison",
        "NIDDK: complete comparison table of all three tests")
    add("test_comparison", "How does the sensitivity of A1C compare to FPG and OGTT?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["FPG > A1C", "OGTT > FPG", "sensitivity ranking"], "medium", "test_comparison",
        "NIDDK: FPG sensitivity > A1C; OGTT sensitivity > FPG")
    add("test_comparison", "What is the sensitivity of the FPG test compared to A1C?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["FPG sensitivity", "greater than A1C"], "easy", "test_comparison",
        "NIDDK: FPG sensitivity greater than A1C test")

    # ==================================================================
    # CATEGORY 16: Table Retrieval (ADA Table 2.1, 2.2)
    # ==================================================================
    add("table_retrieval", "What does Table 2.1 in the ADA standards specify?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["Table 2.1", "diagnostic criteria", "nonpregnant"], "medium", "table_lookup",
        "p2: Table 2.1 - Criteria for diagnosis in nonpregnant individuals")
    add("table_retrieval", "What does Table 2.2 in the ADA standards specify?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes"], [2],
        ["Table 2.2", "prediabetes criteria", "nonpregnant"], "medium", "table_lookup",
        "p2: Table 2.2 - Criteria defining prediabetes in nonpregnant individuals")
    add("table_retrieval", "What is the complete diagnostic threshold table for diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["Table 2.1", "all tests", "A1C", "FPG", "OGTT", "random PG"], "hard", "table_lookup",
        "p2: Table 2.1 with all four tests and their thresholds")

    # ==================================================================
    # CATEGORY 17: Multi-part Questions
    # ==================================================================
    add("multi_part", "What are all the diagnostic tests for diabetes and their thresholds?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["A1C 6.5%", "FPG 126", "OGTT 200", "random PG 200"], "medium", "diagnostic_criteria",
        "p2 Table 2.1: all four tests with thresholds")
    add("multi_part", "What tests are used for prediabetes screening and what are their cutoff values?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes"], [10],
        ["A1C 5.7-6.4%", "FPG 100-125", "2-h PG 140-199"], "medium", "prediabetes_criteria",
        "p10: three tests with prediabetes cutoffs")

    # ==================================================================
    # CATEGORY 18: Multi-hop Questions (require combining info from sections)
    # ==================================================================
    add("multi_hop", "What is the difference between how diabetes is diagnosed and how it is confirmed?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes", "Confirming the Diagnosis"], [2, 4],
        ["diagnostic test", "confirmation", "two abnormal results"], "hard", "diagnosis_confirmation",
        "Requires p2 (diagnostic criteria) and p4 (confirmation process)")
    add("multi_hop", "How do the diagnostic criteria for diabetes differ from those for prediabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes", "Diagnosis of Prediabetes"], [2, 10],
        ["higher thresholds", "diabetes vs prediabetes", "comparison"], "hard", "diagnostic_criteria",
        "Requires comparing Table 2.1 (diabetes) with Table 2.2/p10 (prediabetes)")
    add("multi_hop", "What screening approach should be used for someone with risk factors for both type 2 diabetes and gestational diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults", "Gestational Diabetes Mellitus"], [11, 16],
        ["screening intervals", "risk factors", "GDM screening"], "hard", "screening",
        "Requires combining general screening (p11) with GDM-specific screening (p16)")

    # ==================================================================
    # CATEGORY 19: Ambiguous Questions
    # ==================================================================
    add("ambiguous", "What is the normal blood sugar range?",
        True, ["ada_soc_2026_diagnosis", "niddk_diabetes_prediabetes_tests"], ["Screening and Diagnosis of Diabetes", "Diagnosis of Prediabetes"], [2, 10],
        ["normal range", "prediabetes range", "diabetes range", "context"], "hard", "diagnostic_criteria",
        "Ambiguous: could mean normal, prediabetic, or diabetic ranges")
    add("ambiguous", "When should I get tested for diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults"], [11],
        ["screening criteria", "risk factors", "asymptomatic adults"], "medium", "screening",
        "Ambiguous: depends on individual risk factors")

    # ==================================================================
    # CATEGORY 20: Out-of-scope Questions (must refuse)
    # ==================================================================
    add("unsupported", "What is the best cancer treatment for lung cancer?",
        False, [], [], [],
        [], "easy", "unsupported",
        "Out of scope for diabetes documents")
    add("unsupported", "How does exercise affect insulin resistance?",
        False, [], [], [],
        [], "easy", "unsupported",
        "Not covered in these two documents")
    add("unsupported", "What is the recommended daily carbohydrate intake?",
        False, [], [], [],
        [], "easy", "unsupported",
        "Not covered in diagnosis/classification documents")
    add("unsupported", "What medication should I take for diabetes?",
        False, [], [], [],
        [], "easy", "unsupported",
        "Medication is out of scope for diagnosis documents")
    add("unsupported", "How does metformin work in the body?",
        False, [], [], [],
        [], "easy", "unsupported",
        "Pharmacology not covered in diagnosis documents")

    # ==================================================================
    # CATEGORY 21: Unsupported/Refusal Questions (personal advice)
    # ==================================================================
    add("personal_advice", "My A1C is 6.7. Do I have diabetes?",
        False, [], [], [],
        [], "medium", "unsupported",
        "Requires clinical judgment, not a retrieval question")
    add("personal_advice", "I have a fasting glucose of 110. Is that normal?",
        False, [], [], [],
        [], "medium", "unsupported",
        "Requires clinical interpretation, not a retrieval question")

    # ==================================================================
    # CATEGORY 22: Adversarial Questions (designed to confuse)
    # ==================================================================
    add("adversarial", "What is the FPG threshold for gestational diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Gestational Diabetes Mellitus"], [16, 17],
        ["gestational", "FPG", "glucose threshold"], "hard", "gestational",
        "Tests whether general FPG threshold (126) is confused with gestational thresholds")
    add("adversarial", "What autoantibodies are used to diagnose type 2 diabetes?",
        False, [], [], [],
        [], "hard", "unsupported",
        "Type 2 diabetes is not diagnosed with autoantibodies; this is a type 1 concept")
    add("adversarial", "What is the OGTT threshold for confirming type 1 diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Type 1 Diabetes"], [6, 7],
        ["type 1", "OGTT", "autoantibodies", "staging"], "hard", "type1",
        "Tests whether general OGTT threshold is confused with type 1 staging criteria")

    # ==================================================================
    # CATEGORY 23: Section Confusion Questions
    # ==================================================================
    add("section_confusion", "What is the difference between prediabetes diagnosis and type 1 diabetes staging?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes", "Type 1 Diabetes"], [10, 6],
        ["prediabetes criteria", "type 1 staging", "autoimmunity vs metabolic"], "hard", "diagnostic_criteria",
        "Tests whether prediabetes section is confused with type 1 section")
    add("section_confusion", "How does the confirmation of diabetes diagnosis relate to the screening process?",
        True, ["ada_soc_2026_diagnosis"], ["Confirming the Diagnosis", "Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults"], [4, 11],
        ["confirmation", "screening", "different processes"], "hard", "diagnosis_confirmation",
        "Tests whether confirmation section is confused with screening section")

    # ==================================================================
    # CATEGORY 24: Exact Threshold Retrieval
    # ==================================================================
    add("exact_threshold", "What A1C level in mmol/mol corresponds to the diabetes threshold?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [2],
        ["48 mmol/mol", "6.5%", "A1C"], "easy", "diagnostic_criteria",
        "Exact threshold: 48 mmol/mol = 6.5%")
    add("exact_threshold", "What is the exact FPG threshold in both mg/dL and mmol/L for prediabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes"], [10],
        ["100-125 mg/dL", "5.6-6.9 mmol/L"], "medium", "prediabetes_criteria",
        "Exact dual-unit threshold for IFG")
    add("exact_threshold", "What is the exact 2-hour glucose threshold in mmol/L for impaired glucose tolerance?",
        True, ["ada_soc_2026_diagnosis"], ["Diagnosis of Prediabetes"], [10],
        ["140-199 mg/dL", "7.8-11.0 mmol/L"], "medium", "prediabetes_criteria",
        "Exact dual-unit threshold for IGT")

    # ==================================================================
    # CATEGORY 25: Source-Specific Retrieval
    # ==================================================================
    add("source_specific", "What does the NIDDK say about the pros and cons of the FPG test?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["NIDDK", "FPG pros", "FPG cons"], "medium", "test_comparison",
        "NIDDK-specific content about FPG advantages and disadvantages")
    add("source_specific", "According to the ADA 2026 standards, what are the stages of type 1 diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Type 1 Diabetes"], [6, 7],
        ["ADA", "Stage 1", "Stage 2", "Stage 3"], "medium", "type1",
        "ADA-specific type 1 diabetes staging")
    add("source_specific", "What does the NIDDK comparison table say about OGTT sample stability?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["NIDDK", "OGTT", "sample stability", "low"], "medium", "ogtt",
        "NIDDK-specific: OGTT has low sample stability")
    add("source_specific", "What is the ADA recommendation for screening frequency in at-risk adults?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults"], [11, 12],
        ["ADA", "screening", "every 3 years"], "medium", "screening",
        "ADA-specific: every 3 years for at-risk adults")

    # ==================================================================
    # ADDITIONAL QUESTIONS FOR COVERAGE
    # ==================================================================
    add("a1c", "What is the relationship between A1C and average blood glucose?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [3],
        ["glucose exposure", "120 days", "indirect measure"], "medium", "a1c",
        "p3: A1C is an indirect measure of glucose exposure over ~120 days")
    add("a1c", "What is the recommended proficiency testing frequency for A1C?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [3],
        ["three times per year", "proficiency testing"], "hard", "a1c",
        "p3: proficiency testing at least three times per year")
    add("fpg", "What type of sample is recommended for the FPG test?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["sodium fluoride plasma", "serum not recommended"], "hard", "fpg",
        "NIDDK: sodium fluoride plasma preferred; many labs measure serum which is not recommended")
    add("ogtt", "Why is OGTT considered the most sensitive test for diabetes?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["most sensitive", "early marker", "impaired glucose metabolism"], "medium", "test_comparison",
        "NIDDK: OGTT has greater sensitivity than FPG and A1C")
    add("type1", "What is the role of HLA haplotype in type 1 diabetes risk?",
        True, ["ada_soc_2026_diagnosis"], ["Type 1 Diabetes"], [7, 8],
        ["HLA", "genetic risk", "DR", "DQ"], "hard", "type1",
        "p7-8: HLA haplotype contributes to genetic risk for type 1 diabetes")
    add("type1", "What are the criteria for Stage 3 type 1 diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Type 1 Diabetes"], [6, 7],
        ["clinical diabetes", "hyperglycemia", "symptomatic"], "hard", "type1",
        "p6-7: Stage 3 is clinical diabetes with symptoms")
    add("gestational", "What is the two-step approach for gestational diabetes screening?",
        True, ["ada_soc_2026_diagnosis"], ["Gestational Diabetes Mellitus"], [16, 17],
        ["50-g GLT", "100-g OGTT", "two-step"], "hard", "gestational",
        "p16-17: Step 1: 50-g GLT; Step 2: 100-g OGTT if positive")
    add("gestational", "When should gestational diabetes screening begin?",
        True, ["ada_soc_2026_diagnosis"], ["Gestational Diabetes Mellitus"], [16],
        ["24-28 weeks", "earlier if high risk"], "medium", "gestational",
        "p16: 24-28 weeks; earlier for high-risk individuals")
    add("monogenic", "What genetic tests are recommended for suspected monogenic diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Monogenic Diabetes Syndromes"], [15, 16],
        ["HNF1A", "GCK", "KCNJ11", "genetic testing"], "hard", "monogenic",
        "p15-16: genetic testing for specific MODY genes")
    add("special_populations", "What are the unique considerations for diabetes diagnosis in pregnancy?",
        True, ["ada_soc_2026_diagnosis"], ["Gestational Diabetes Mellitus"], [16, 17],
        ["pregnancy-specific", "GDM", "IADPSG", "glucose thresholds"], "medium", "gestational",
        "p16-17: pregnancy-specific diagnostic criteria")
    add("diagnostic_criteria", "What is the role of the DCCT in A1C standardization?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Diagnosis of Diabetes"], [3],
        ["DCCT", "NGSP", "standardization", "reference assay"], "hard", "diagnostic_criteria",
        "p3: A1C standardized to DCCT reference assay through NGSP")
    add("test_comparison", "What is the within-patient variability of each diabetes test?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["FPG variability", "A1C low variability", "OGTT high variability"], "hard", "test_comparison",
        "NIDDK: A1C has low within-patient variability; FPG and OGTT have higher")
    add("test_comparison", "How does sample stability differ between FPG, OGTT, and A1C?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["FPG low", "OGTT low", "A1C superior"], "medium", "test_comparison",
        "NIDDK: A1C has superior sample stability; FPG and OGTT are low")
    add("test_comparison", "What is the cost comparison between diabetes diagnostic tests?",
        True, ["niddk_diabetes_prediabetes_tests"], ["Comparing Diabetes Blood Tests"], [1],
        ["FPG low cost", "OGTT higher cost", "A1C higher cost"], "medium", "test_comparison",
        "NIDDK: FPG is low cost; OGTT and A1C are higher cost")
    add("diagnostic_criteria", "What is the clinical significance of having one vs two abnormal test results?",
        True, ["ada_soc_2026_diagnosis"], ["Confirming the Diagnosis"], [4],
        ["single abnormal", "two abnormal", "confirmation", "diagnostic certainty"], "hard", "diagnosis_confirmation",
        "p4: single abnormal requires confirmation; two abnormal from different tests is diagnostic")
    add("screening", "Should all pregnant women be screened for diabetes?",
        True, ["ada_soc_2026_diagnosis"], ["Gestational Diabetes Mellitus"], [16],
        ["universal screening", "GDM", "pregnancy"], "medium", "gestational",
        "p16: screening recommended for all pregnant individuals at 24-28 weeks")
    add("screening", "What is the recommendation for diabetes screening in HIV patients?",
        True, ["ada_soc_2026_diagnosis"], ["Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults"], [11, 12],
        ["HIV", "increased risk", "screening"], "hard", "screening",
        "p11: HIV is a risk factor that may warrant more frequent screening")

    # Save
    output_path = Path("backend/app/evaluation/test_set_v3.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    # Also save V2 test set for reference
    import shutil
    existing_path = Path("backend/app/evaluation/test_questions.json")
    v2_path = Path("backend/app/evaluation/test_set_v2.json")
    shutil.copy2(existing_path, v2_path)

    # Statistics
    answerable = sum(1 for q in questions if q["answerable"])
    unanswerable = len(questions) - answerable

    categories = {}
    difficulties = {"easy": 0, "medium": 0, "hard": 0}
    intents = {}
    for q in questions:
        categories[q["category"]] = categories.get(q["category"], 0) + 1
        difficulties[q["difficulty"]] = difficulties.get(q["difficulty"], 0) + 1
        intents[q["intent"]] = intents.get(q["intent"], 0) + 1

    print(f"V3 Test Set Created: {len(questions)} questions")
    print(f"  Answerable: {answerable}")
    print(f"  Unanswerable (must refuse): {unanswerable}")
    print(f"\nBy difficulty:")
    for d, c in sorted(difficulties.items()):
        print(f"  {d}: {c}")
    print(f"\nBy category:")
    for c, n in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")
    print(f"\nBy intent:")
    for i, n in sorted(intents.items(), key=lambda x: -x[1]):
        print(f"  {i}: {n}")
    print(f"\nSaved to: {output_path}")
    print(f"V2 preserved at: {v2_path}")

    return questions


if __name__ == "__main__":
    build_v3_test_set()
