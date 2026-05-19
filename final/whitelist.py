"""
Whitelist configuration for user IDs that should always be marked as GENUINE.
Add or remove user IDs from the WHITELISTED_IDS list as needed.
"""

# List of user IDs that will always be marked as GENUINE
WHITELISTED_IDS = [
    '12345678',         # Example user
    '87654321',         # Example user
    'user123',          # Example user
    'influencer1',      # Example user
    'verified_user',    # Example user
    'sayeedahamed_0703' # Specific user to be whitelisted
]

def is_whitelisted(user_id):
    """Check if a user ID is in the whitelist"""
    return str(user_id).strip() in WHITELISTED_IDS
