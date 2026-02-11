"""Constants for categories and subcategories."""

INCOME_CATEGORIES = [
    "Salary",
    "Bonus",
    "Interest",
    "Refund",
    "Other Income",
]

EXPENSE_CATEGORIES = [
    "Fixed Expenses",
    "Groceries",
    "Shopping",
    "Eating Out",
    "Entertainment",
    "Transportation",
    "Health",
]

CATEGORY_TO_SUBCATEGORIES = {
    "Salary": ["Base Salary", "Commission", "Overtime", "Other"],
    "Bonus": ["Performance", "Sign-on", "Referral", "Other"],
    "Interest": ["Bank Interest", "Investments", "Other"],
    "Refund": ["Purchase Refund", "Tax Refund", "Other"],
    "Other Income": ["Gift", "Cashback", "Other"],
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
        "Pizza",
        "Burger",
        "Other",
    ],
    "Entertainment": ["Cinema", "Tourism", "Events", "Hotel"],
    "Transportation": ["Fuel", "Parking", "Maintenance", "Rideshare", "Other"],
    "Health": ["Pharmacy", "Doctor/Dentist", "Exams", "Other"],
}
