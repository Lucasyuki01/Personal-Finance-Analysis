"""Default categories and subcategories."""

DEFAULT_INCOME_TAXONOMY = {
    "Salary": ["Base Salary", "Commission", "Overtime", "Other"],
    "Bonus": ["Performance", "Sign-on", "Referral", "Other"],
    "Interest": ["Bank Interest", "Investments", "Other"],
    "Refund": ["Purchase Refund", "Tax Refund", "Other"],
    "Other Income": ["Gift", "Cashback", "Other"],
}

DEFAULT_EXPENSE_TAXONOMY = {
    "Fixed Expenses": ["Rent", "Phone", "Insurance", "Gym", "Utilities", "Subscriptions", "Transit", "Other fixed"],
    "Groceries": ["Supermarket", "Convenience", "Liquor", "Wholesale"],
    "Shopping": ["Electronics", "Clothing", "Home", "General"],
    "Eating Out": [
        "Fast Food",
        "Cafe",
        "Delivery",
        "Bar",
        "Cuisine: Japanese",
        "Cuisine: Korean",
        "Cuisine: Arabic",
        "Cuisine: Latin",
        "Cuisine: Chinese",
        "Cuisine: Canadian",
        "Pizza",
        "Burger",
        "Vending Machine",
        "Dessert",
        "Other",
    ],
    "Entertainment": ["Cinema", "Tourism", "Events", "Hotel", "Other"],
    "Transportation": ["Fuel", "Parking", "Maintenance", "Rideshare", "Other"],
    "Health": ["Pharmacy", "Doctor/Dentist", "Exams", "Other"],
    "Services": ["Other"],
    "Other": ["Other"],
}

DEFAULT_INCOME_CATEGORIES = list(DEFAULT_INCOME_TAXONOMY.keys())
DEFAULT_EXPENSE_CATEGORIES = list(DEFAULT_EXPENSE_TAXONOMY.keys())

DEFAULT_INCOME_SUBCATEGORIES = DEFAULT_INCOME_TAXONOMY
DEFAULT_EXPENSE_SUBCATEGORIES = DEFAULT_EXPENSE_TAXONOMY

DEFAULT_CATEGORY_TO_SUBCATEGORIES = {**DEFAULT_INCOME_TAXONOMY, **DEFAULT_EXPENSE_TAXONOMY}
