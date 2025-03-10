// Connect to the default database (MongoDB creates it automatically)
db = db.getSiblingDB("reddit_sentiment");

// Create collections if they don't exist
db.createCollection("posts");
db.createCollection("comments");

// Create unique indexes to prevent duplicate entries
db.posts.createIndex({ id: 1 }, { unique: true });
db.comments.createIndex({ id: 1 }, { unique: true });

print("✅ MongoDB initialized with 'reddit_sentiment' database and collections.");
