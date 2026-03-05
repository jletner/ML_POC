# Vector-Based Budget Categorizer

A semantic vector embedding approach to categorizing budget transactions. This uses OpenAI's `text-embedding-3-small` model to understand the meaning of merchant names and match them to categories.

## How It Works

1. **Training**: Creates embeddings for each category by combining:
   - Category descriptions (e.g., "coffee shop, cafe, espresso...")
   - Sample merchant names from training data

2. **Classification**: When a new transaction comes in:
   - Generate embedding for the merchant name
   - Calculate cosine similarity to all category embeddings
   - Return the most similar category

## Advantages Over Bag-of-Words

| Feature                  | Bag of Words | Vector Embeddings              |
| ------------------------ | ------------ | ------------------------------ |
| Handles unseen merchants | ❌ Poor      | ✅ Good                        |
| Semantic understanding   | ❌ None      | ✅ "coffee shop" ≈ "Starbucks" |
| Training data needed     | More         | Less                           |
| Speed                    | Fast         | Requires API call              |
| Cost                     | Free         | OpenAI API costs               |

## Setup

1. Install dependencies:

   ```bash
   cd vector_solution
   npm install
   ```

2. Set your OpenAI API key:

   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY = "sk-..."

   # Or add to .env file
   ```

3. Train embeddings:

   ```bash
   npm run train
   ```

4. Start server:
   ```bash
   npm run serve
   ```

## API Endpoints

### Classify Transaction

```
GET /classify?merchant=Starbucks&amount=5.50
```

Response:

```json
{
  "merchant": "Starbucks",
  "amount": 5.5,
  "prediction": "Coffee & Drinks",
  "confidence": 0.89,
  "matchType": "vector",
  "topMatches": [
    { "category": "Coffee & Drinks", "confidence": 0.89 },
    { "category": "Food & Dining", "confidence": 0.72 }
  ]
}
```

### Save Correction

```
POST /correct
Content-Type: application/json

{"merchant": "Local Cafe", "category": "Coffee & Drinks"}
```

### List Categories

```
GET /categories
```

### Health Check

```
GET /health
```

## File Structure

```
vector_solution/
├── package.json
├── train-embeddings.ts    # Generates category embeddings
├── vector-server.ts       # HTTP server for classification
├── README.md
└── embeddings/            # Generated after training
    ├── category-embeddings.json
    ├── merchant-lookup.json
    └── categories.json
```

## Cost Estimation

Using `text-embedding-3-small`:

- Training: ~$0.01 for ~100 merchants
- Per classification: ~$0.00002 (practically free)
- 10,000 classifications ≈ $0.20
