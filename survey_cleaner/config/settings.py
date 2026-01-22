"""Configuration and constants for the survey cleaner app."""

# Clustering settings
DEFAULT_SIMILARITY_THRESHOLD = 90
MIN_SIMILARITY_THRESHOLD = 80
MAX_SIMILARITY_THRESHOLD = 100

# Age validation
MIN_VALID_AGE = 20
MAX_VALID_AGE = 100

# Text replacements for normalization
ABBREVIATION_REPLACEMENTS = {
    r'\basst\.?\b': 'assistant',
    r'\bassst\.?\b': 'assistant',
    r'\basssitant\b': 'assistant',
    r'\bassitant\b': 'assistant',
    r'\bassoc\.?\b': 'associate',
    r'\bvist\.?\b': 'visiting',
    r'\bvisit\.?\b': 'visiting',
    r'\bprof\.?\b': 'professor',
    r'\bprofesso\b': 'professor',
    r'\bprofesor\b': 'professor',
    r'\bdept\.?\b': 'department',
    r'\bdir\.?\b': 'director',
    r'\bmgr\.?\b': 'manager',
}

POSTDOC_REPLACEMENTS = {
    r'\bpos-doc\b': 'postdoctoral',
    r'\bpost-doc\b': 'postdoctoral',
    r'\bpostdoc\b': 'postdoctoral',
    r'\bpstdoctoral\b': 'postdoctoral',
    r'\bpostdoctral\b': 'postdoctoral',
    r'\bposdoctoral\b': 'postdoctoral',
    r'\bpost doctoral\b': 'postdoctoral',
    r'\bpostdoctor\b': 'postdoctoral',
    r'\bpostdoctorate\b': 'postdoctoral',
}

COMMON_TYPO_REPLACEMENTS = {
    r'\breserch\b': 'research',
    r'\bresearch\b': 'research',
    r'\brsearch\b': 'research',
    r'\bscientis\b': 'scientist',
    r'\blecture\b': 'lecturer',
    r'\bintructor\b': 'instructor',
    r'\binstrutor\b': 'instructor',
    r'\binsructor\b': 'instructor',
    r'\bfello\b': 'fellow',
}

# Job title keywords for normalization (ordered by specificity)
JOB_KEYWORDS = [
    "visiting lecturer", "visiting assistant professor", 
    "visiting associate professor", "visiting professor",
    "visiting instructor",
    "adjunct lecturer", "adjunct assistant professor", 
    "adjunct associate professor", "adjunct professor",
    "adjunct instructor",
    "acting assistant professor", "acting associate professor", 
    "acting professor", "acting instructor",
    "temporary assistant professor", "temporary associate professor", 
    "temporary professor", "temporary instructor",
    "clinical assistant professor", "clinical associate professor", 
    "clinical professor", "clinical instructor",
    "research assistant professor", "research associate professor", 
    "research professor", "research instructor",
    "lecturer", "assistant professor", "associate professor", 
    "professor", "instructor",
    "postdoctoral",
    "research scientist", "research fellow", "research associate", 
    "research assistant",
    "director", "associate director", "assistant director",
    "biostatistician", "statistician",
]

# Job title detection keywords
JOB_TITLE_KEYWORDS = [
    'title', 'position', 'job', 'role', 'designation', 'Emp_Position_Title'
]

JOB_CONTENT_KEYWORDS = [
    'professor', 'lecturer', 'assistant', 'associate', 'director', 
    'manager', 'specialist', 'coordinator', 'analyst', 'engineer'
]
