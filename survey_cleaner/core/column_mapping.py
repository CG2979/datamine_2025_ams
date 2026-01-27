"""Column mapping utilities for merging files with different column names."""

import pandas as pd
from rapidfuzz import fuzz
from typing import Dict, List, Tuple


def suggest_column_mappings(cols1: List[str], cols2: List[str], threshold: int = 80) -> Dict[str, str]:
    """
    Suggest column mappings between two sets of column names using fuzzy matching.
    
    Args:
        cols1: Column names from first file
        cols2: Column names from second file
        threshold: Minimum similarity score (0-100) for auto-suggestion
    
    Returns:
        Dictionary mapping cols2 to cols1: {col2_name: col1_name}
    """
    suggestions = {}
    
    for col2 in cols2:
        if col2 in cols1:
            # Exact match - map to itself
            suggestions[col2] = col2
            continue
        
        # Find best fuzzy match
        best_match = None
        best_score = 0
        
        for col1 in cols1:
            score = fuzz.ratio(col2.lower(), col1.lower())
            if score > best_score and score >= threshold:
                best_score = score
                best_match = col1
        
        if best_match:
            suggestions[col2] = best_match
    
    return suggestions


def get_common_semantic_mappings() -> Dict[str, List[str]]:
    """
    Return common semantic equivalents for column names.
    Uses actual column names from doctoral survey data as canonical references.
    
    Returns:
        Dictionary mapping canonical names to list of alternatives
    """
    return {
        # Identity fields - canonical: Recipient_Number
        "Recipient_Number": ["recipient_code", "serial_number", "employee_id", "emp_id", "employee_number", "emp_no", "staff_id", "worker_id", "id", "person_id"],
        "Serial_Number_Granting_Dept": ["serial_number", "dept_serial", "department_id"],
        "Citizen_Name_DG": ["name", "full_name", "person_name", "citizen_name"],
        
        # Job/Position fields - canonical: Emp_Position_Title
        "Emp_Position_Title": ["job_title", "title", "position", "role", "job_position", "job", "position_title", "employment_title"],
        "Employment_Type": ["emp_type", "empl_type", "employment_status", "emp_status", "empl_type_calc", "2012emp_type", "work_type"],
        "Emp_Status_Code_DG": ["emp_status_code", "employment_status_code", "status_code"],
        "Emp_Status_Name_DG": ["emp_status_name", "employment_status", "emp_status", "status_name"],
        "Empl_Matrix_Code_DG": ["empl_matrix_code", "matrix_code", "classification_code", "empl_matrix_code_old"],
        "Empl_Matrix_Name_DG": ["empl_matrix_name", "matrix_name", "classification_name"],
        "Emp_Division_Department": ["department", "dept", "division", "unit", "section", "degree_department"],
        
        # Institution/Employer fields - canonical: Empl_Inst_Org
        "Empl_Inst_Org": ["employer_name", "employment_institution", "employer", "organization", "institution"],
        "Employment_Institution": ["employer", "organization", "empl_inst_org", "institution_name"],
        "Degree Institution": ["degree_institution", "university", "school", "grad_institution"],
        "Fall_Institution": ["fall_inst", "fall_school"],
        "Summer_Institution": ["summer_inst", "summer_school"],
        
        # Department fields - canonical: Degree Department
        "Degree Department": ["degree_dept", "major_dept", "grad_department"],
        "Fall_Dept": ["fall_department", "fall_dept_name"],
        "Summer_Dept": ["summer_department", "summer_dept_name"],
        
        # Location fields - canonical: Empl_City, Empl_State, etc.
        "Empl_City": ["city", "employment_city", "work_city"],
        "Empl_State": ["state", "employment_state", "work_state"],
        "Empl_Country_Calc": ["empl_country", "country", "employment_country", "work_country"],
        "Empl_Zip": ["zip", "zipcode", "postal_code", "zip_code", "postcode"],
        "City": ["empl_city", "location_city", "residence_city"],
        "State": ["empl_state", "location_state", "residence_state"],
        "Country": ["empl_country", "location_country", "residence_country"],
        "Zip": ["empl_zip", "zipcode", "postal_code"],
        "Fall_State": ["fall_location_state"],
        "Summer_State": ["summer_location_state"],
        "Fall_Country": ["fall_location_country"],
        "Summer_Country": ["summer_location_country"],
        
        # Personal fields - canonical: Age, Gender Code
        "Age": ["years_old", "employee_age", "age_years"],
        "Gender Code": ["gender", "sex", "gender_code"],
        "Citizenship": ["citizenship_type", "nationality", "citizen"],
        "Citizenship_Type": ["citizenship", "citizen_type"],
        
        # Compensation - canonical: Salary
        "Salary": ["compensation", "pay", "wage", "income", "annual_salary"],
        
        # Dates - canonical: Date_Conferred_Code
        "Date_Conferred_Code": ["date_conferred", "degree_date", "graduation_date", "conferral_date"],
        
        # Survey-specific fields - canonical from your data
        "Survey_Code": ["surcode", "survey_id"],
        "Survey_Code_2": ["surcode_2", "survey_id_2", "alternate_survey_code"],
        "Survey_Group": ["surgroup", "survey_category", "group"],
        "Surcode": ["survey_code", "survey_id"],
        "Surcode_2": ["survey_code_2"],
        "Surgroup": ["survey_group", "group_code"],
        "Surgroup_2": ["survey_group_2"],
        "survey_year": ["source_year", "year", "data_year", "survey_yr"],
        "source_year": ["survey_year", "year", "data_year"],
        "survey_type": ["type", "survey_category"],
        "2012Survey_Group": ["survey_group", "surgroup"],
        
        # Status fields - canonical from your data
        "Tenured": ["tenure_status", "tenure", "is_tenured"],
        "Fulltime": ["full_time", "ft", "full_time_status", "fulltime_status"],
        "Part_Time_Code_DG": ["part_time_code", "pt_code", "parttime_code"],
        "Part_Time_Name_DG": ["part_time_name", "pt_status", "parttime_status"],
        "Permanent": ["permanent_status", "permanent_position", "is_permanent"],
        "Postdoctorate_Code_DG": ["postdoc_code", "postdoctoral_code", "postdoc_status_code"],
        "Postdoctorate_Name_DG": ["postdoc_name", "postdoctoral_status", "postdoc_status"],
        
        # Job seeking status - canonical from your data
        "Seeking": ["currently_looking", "job_seeking", "looking_for_work", "seeking_employment"],
        "Currently looking": ["seeking", "job_seeking", "looking_for_work"],
        "Working": ["employed", "currently_working", "is_working", "employment_status"],
        
        # Degree fields - canonical from your data
        "Type_of_Degree_DG": ["type_of_degree", "degree_type", "degree"],
        "MR Field": ["mr_field", "major", "field_of_study", "degree_field"],
        "MR_Code_DG": ["mr_code", "major_code", "field_code"],
        
        # Employment matrix - canonical from your data
        "Empl_Matrix_Code": ["empl_matrix_code_dg", "matrix_code", "classification_code"],
        "Empl_Matrix_Name": ["empl_matrix_name_dg", "matrix_name", "classification_name"],
        "Empl_Matrix_Code_OLD": ["old_matrix_code", "previous_matrix_code"],
        
        # Visa/Immigration - canonical from your data
        "Visa_Code_DG": ["visa_code", "visa_status", "immigration_status"],
        "Left_US_Code_DG": ["left_us_code", "departed_us_code"],
        "Left_US_DG": ["left_us", "departed", "left_country"],
        
        # Counts - canonical from your data
        "Count_Cit_Female": ["count_female", "female_count", "num_female", "female_citizens"],
        "Count_Cit_Male": ["count_male", "male_count", "num_male", "male_citizens"],
        
        # Forms/Records - canonical from your data
        "DG_Forms_Current_Yr": ["dg_forms", "forms_current_year", "current_forms"],
        "DG_No_Forms": ["dg_number_forms", "num_forms", "form_count"],
        
        # Employment type variations - canonical from your data
        "2012Emp_Type": ["emp_type", "employment_type", "empl_type"],
        "Empl_Type_Calc": ["employment_type", "emp_type_calculated", "calculated_emp_type"],
        "Employment_Status": ["emp_status", "empl_status", "employment_stat"],
        
        # Email and contact
        "Email": ["email_address", "e-mail", "mail", "email_id"],
        "Phone": ["phone_number", "telephone", "tel", "mobile", "contact_number"],
        
        # Other specific fields
        "Employer name": ["employer_name", "empl_name", "employer"],
        "Moved to EENDR Database": ["moved_to_eendr", "eendr_flag", "in_eendr"],
    }


def get_year_pattern_mappings(cols1: List[str], cols2: List[str]) -> Dict[str, str]:
    """
    Detect and suggest mappings for year-prefixed columns (e.g., 2012Emp_Type → 2013Emp_Type).
    
    Args:
        cols1: Column names from first file
        cols2: Column names from second file
    
    Returns:
        Dictionary mapping cols2 to cols1: {col2_name: col1_name}
    """
    import re
    
    suggestions = {}
    
    # Pattern to match year prefix (e.g., "2012Emp_Type")
    year_pattern = re.compile(r'^(\d{4})(.+)$')


def suggest_semantic_mappings(cols1: List[str], cols2: List[str]) -> Dict[str, str]:
    """
    Suggest mappings based on semantic equivalents.
    
    Args:
        cols1: Column names from first file
        cols2: Column names from second file
    
    Returns:
        Dictionary mapping cols2 to cols1: {col2_name: col1_name}
    """
    semantic_map = get_common_semantic_mappings()
    suggestions = {}
    
    # Create reverse lookup: alternative -> canonical
    alt_to_canonical = {}
    for canonical, alternatives in semantic_map.items():
        for alt in alternatives:
            alt_to_canonical[alt.lower()] = canonical
        alt_to_canonical[canonical.lower()] = canonical
    
    # Map cols1 to their canonical forms
    cols1_canonical = {}
    for col1 in cols1:
        canonical = alt_to_canonical.get(col1.lower(), col1.lower())
        cols1_canonical[canonical] = col1
    
    # Map cols2 to cols1 via canonical forms
    for col2 in cols2:
        if col2 in cols1:
            suggestions[col2] = col2
            continue
            
        canonical = alt_to_canonical.get(col2.lower())
        if canonical and canonical in cols1_canonical:
            suggestions[col2] = cols1_canonical[canonical]
    
    # Also try year pattern matching
    year_suggestions = get_year_pattern_mappings(cols1, cols2)
    suggestions.update(year_suggestions)
    
    return suggestions


def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Apply column mapping to a dataframe (rename columns).
    
    Args:
        df: Input dataframe
        mapping: Dictionary mapping old names to new names
    
    Returns:
        DataFrame with renamed columns
    """
    df_copy = df.copy()
    
    # Only rename columns that exist in the dataframe
    valid_mapping = {old: new for old, new in mapping.items() if old in df_copy.columns}
    
    df_copy = df_copy.rename(columns=valid_mapping)
    
    return df_copy


def get_unmapped_columns(cols1: List[str], cols2: List[str], mapping: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Get columns that haven't been mapped yet.
    
    Args:
        cols1: Column names from first file
        cols2: Column names from second file
        mapping: Current mapping dict {col2: col1}
    
    Returns:
        Tuple of (unmapped_from_cols1, unmapped_from_cols2)
    """
    mapped_cols1 = set(mapping.values())
    mapped_cols2 = set(mapping.keys())
    
    unmapped_cols1 = [col for col in cols1 if col not in mapped_cols1]
    unmapped_cols2 = [col for col in cols2 if col not in mapped_cols2]
    
    return unmapped_cols1, unmapped_cols2


def validate_mapping(mapping: Dict[str, str], cols1: List[str], cols2: List[str]) -> List[str]:
    """
    Validate a column mapping and return list of errors.
    
    Args:
        mapping: Proposed mapping {col2: col1}
        cols1: Valid column names from file 1
        cols2: Valid column names from file 2
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    for col2, col1 in mapping.items():
        if col2 not in cols2:
            errors.append(f"Source column '{col2}' does not exist in File 2")
        if col1 not in cols1:
            errors.append(f"Target column '{col1}' does not exist in File 1")
    
    # Check for duplicate mappings (multiple col2 -> same col1)
    target_counts = {}
    for col2, col1 in mapping.items():
        target_counts[col1] = target_counts.get(col1, 0) + 1
    
    for col1, count in target_counts.items():
        if count > 1:
            sources = [col2 for col2, target in mapping.items() if target == col1]
            errors.append(f"Multiple columns map to '{col1}': {sources}")
    
    return errors
)
    
    # Build a lookup of base names in cols1
    cols1_base_to_full = {}
    for col1 in cols1:
        match = year_pattern.match(col1)
        if match:
            base_name = match.group(2)
            cols1_base_to_full[base_name.lower()] = col1
    
    # Try to match cols2 to cols1 by base name
    for col2 in cols2:
        if col2 in cols1:
            continue
            
        match = year_pattern.match(col2)
        if match:
            base_name = match.group(2)
            if base_name.lower() in cols1_base_to_full:
                suggestions[col2] = cols1_base_to_full[base_name.lower()]
    
    return suggestions


def suggest_semantic_mappings(cols1: List[str], cols2: List[str]) -> Dict[str, str]:
    """
    Suggest mappings based on semantic equivalents.
    
    Args:
        cols1: Column names from first file
        cols2: Column names from second file
    
    Returns:
        Dictionary mapping cols2 to cols1: {col2_name: col1_name}
    """
    semantic_map = get_common_semantic_mappings()
    suggestions = {}
    
    # Create reverse lookup: alternative -> canonical
    alt_to_canonical = {}
    for canonical, alternatives in semantic_map.items():
        for alt in alternatives:
            alt_to_canonical[alt.lower()] = canonical
        alt_to_canonical[canonical.lower()] = canonical
    
    # Map cols1 to their canonical forms
    cols1_canonical = {}
    for col1 in cols1:
        canonical = alt_to_canonical.get(col1.lower(), col1.lower())
        cols1_canonical[canonical] = col1
    
    # Map cols2 to cols1 via canonical forms
    for col2 in cols2:
        if col2 in cols1:
            suggestions[col2] = col2
            continue
            
        canonical = alt_to_canonical.get(col2.lower())
        if canonical and canonical in cols1_canonical:
            suggestions[col2] = cols1_canonical[canonical]
    
    return suggestions


def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Apply column mapping to a dataframe (rename columns).
    
    Args:
        df: Input dataframe
        mapping: Dictionary mapping old names to new names
    
    Returns:
        DataFrame with renamed columns
    """
    df_copy = df.copy()
    
    # Only rename columns that exist in the dataframe
    valid_mapping = {old: new for old, new in mapping.items() if old in df_copy.columns}
    
    df_copy = df_copy.rename(columns=valid_mapping)
    
    return df_copy


def get_unmapped_columns(cols1: List[str], cols2: List[str], mapping: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Get columns that haven't been mapped yet.
    
    Args:
        cols1: Column names from first file
        cols2: Column names from second file
        mapping: Current mapping dict {col2: col1}
    
    Returns:
        Tuple of (unmapped_from_cols1, unmapped_from_cols2)
    """
    mapped_cols1 = set(mapping.values())
    mapped_cols2 = set(mapping.keys())
    
    unmapped_cols1 = [col for col in cols1 if col not in mapped_cols1]
    unmapped_cols2 = [col for col in cols2 if col not in mapped_cols2]
    
    return unmapped_cols1, unmapped_cols2


def validate_mapping(mapping: Dict[str, str], cols1: List[str], cols2: List[str]) -> List[str]:
    """
    Validate a column mapping and return list of errors.
    
    Args:
        mapping: Proposed mapping {col2: col1}
        cols1: Valid column names from file 1
        cols2: Valid column names from file 2
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    for col2, col1 in mapping.items():
        if col2 not in cols2:
            errors.append(f"Source column '{col2}' does not exist in File 2")
        if col1 not in cols1:
            errors.append(f"Target column '{col1}' does not exist in File 1")
    
    # Check for duplicate mappings (multiple col2 -> same col1)
    target_counts = {}
    for col2, col1 in mapping.items():
        target_counts[col1] = target_counts.get(col1, 0) + 1
    
    for col1, count in target_counts.items():
        if count > 1:
            sources = [col2 for col2, target in mapping.items() if target == col1]
            errors.append(f"Multiple columns map to '{col1}': {sources}")
    
    return errors
