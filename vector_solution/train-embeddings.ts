import OpenAI from "openai";
import * as fs from "node:fs";

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Base categories with descriptive text for better embeddings
const CATEGORY_DESCRIPTIONS: Record<string, string> = {
  "Coffee & Drinks":
    "coffee shop, cafe, espresso, latte, tea, beverages, Starbucks, Dunkin, Peets",
  "Food & Dining":
    "restaurant, fast food, dining, meals, takeout, delivery, burger, pizza, tacos",
  Groceries:
    "supermarket, grocery store, food shopping, produce, Kroger, Walmart, Whole Foods, Costco",
  Transportation:
    "gas station, fuel, uber, lyft, taxi, transit, parking, toll, commute",
  Entertainment:
    "streaming, Netflix, Spotify, movies, games, concerts, theater, fun, recreation",
  Shopping:
    "retail, clothing, electronics, Amazon, Target, department store, online shopping",
  "Bills & Utilities":
    "electric, water, gas bill, internet, phone, rent, mortgage, insurance, subscription",
  Other: "miscellaneous, general, uncategorized, various, other expenses",
  Pets: "pet store, veterinary, grooming, pet food, dog, cat, PetSmart, Petco, animal care, pet supplies",
};

interface Transaction {
  merchant: string;
  amount: number;
  category: string;
}

interface CategoryEmbedding {
  category: string;
  embedding: number[];
  merchantExamples: string[];
}

// Get embedding from OpenAI
async function getEmbedding(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    dimensions: 512,
  });
  return response.data[0].embedding;
}

// Batch get embeddings (more efficient)
async function getEmbeddings(texts: string[]): Promise<number[][]> {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: texts,
    dimensions: 512,
  });
  return response.data.map((d: { embedding: number[] }) => d.embedding);
}

// Average multiple embeddings
function averageEmbeddings(embeddings: number[][]): number[] {
  if (embeddings.length === 0) return [];
  const dims = embeddings[0].length;
  const avg = new Array(dims).fill(0);

  for (const emb of embeddings) {
    for (let i = 0; i < dims; i++) {
      avg[i] += emb[i];
    }
  }

  for (let i = 0; i < dims; i++) {
    avg[i] /= embeddings.length;
  }

  // Normalize
  const norm = Math.sqrt(avg.reduce((sum, v) => sum + v * v, 0));
  return avg.map((v) => v / norm);
}

async function train() {
  console.log("Loading training data...");

  // Load training data
  const syntheticData: Transaction[] = JSON.parse(
    fs.readFileSync("./data/synthetic-transactions.json", "utf-8"),
  );
  const userCorrections: Transaction[] = JSON.parse(
    fs.readFileSync("./data/user-corrections.json", "utf-8"),
  );

  // Merge data - user corrections override
  const allData = [...syntheticData, ...userCorrections];

  // Group merchants by category
  const categoryMerchants = new Map<string, Set<string>>();

  for (const t of allData) {
    if (!categoryMerchants.has(t.category)) {
      categoryMerchants.set(t.category, new Set());
    }
    categoryMerchants.get(t.category)!.add(t.merchant);
  }

  // Add any base categories not in data
  for (const cat of Object.keys(CATEGORY_DESCRIPTIONS)) {
    if (!categoryMerchants.has(cat)) {
      categoryMerchants.set(cat, new Set());
    }
  }

  console.log(`Found ${categoryMerchants.size} categories`);

  // Create embeddings for each category
  const categoryEmbeddings: CategoryEmbedding[] = [];

  for (const [category, merchants] of categoryMerchants) {
    console.log(
      `Processing category: ${category} (${merchants.size} merchants)`,
    );

    const merchantList = Array.from(merchants);

    // Create rich description combining:
    // 1. Category description (if available)
    // 2. Sample merchants
    const description = CATEGORY_DESCRIPTIONS[category] || category;
    const merchantSamples = merchantList.slice(0, 10).join(", ");
    const fullText = `${category}: ${description}. Examples: ${merchantSamples}`;

    // Get embedding for category description
    const descEmbedding = await getEmbedding(fullText);

    // Also get embeddings for individual merchants and average
    let finalEmbedding: number[];

    if (merchantList.length > 0) {
      // Get merchant embeddings in batches
      const batchSize = 20;
      const merchantEmbeddings: number[][] = [];

      for (let i = 0; i < Math.min(merchantList.length, 50); i += batchSize) {
        const batch = merchantList.slice(i, i + batchSize);
        const batchEmbs = await getEmbeddings(batch);
        merchantEmbeddings.push(...batchEmbs);
      }

      // Combine description embedding with merchant embeddings
      // Weight description 2x to give it more influence
      const allEmbs = [descEmbedding, descEmbedding, ...merchantEmbeddings];
      finalEmbedding = averageEmbeddings(allEmbs);
    } else {
      finalEmbedding = descEmbedding;
    }

    categoryEmbeddings.push({
      category,
      embedding: finalEmbedding,
      merchantExamples: merchantList.slice(0, 5),
    });
  }

  // Build merchant lookup from user corrections (exact matches)
  const merchantLookup: Record<string, string> = {};
  for (const c of userCorrections) {
    merchantLookup[c.merchant.toLowerCase()] = c.category;
  }

  // Save embeddings
  fs.mkdirSync("./embeddings", { recursive: true });

  fs.writeFileSync(
    "./embeddings/category-embeddings.json",
    JSON.stringify(categoryEmbeddings, null, 2),
  );

  fs.writeFileSync(
    "./embeddings/merchant-lookup.json",
    JSON.stringify(merchantLookup, null, 2),
  );

  fs.writeFileSync(
    "./embeddings/categories.json",
    JSON.stringify(
      categoryEmbeddings.map((c) => c.category),
      null,
      2,
    ),
  );

  console.log("\n✅ Training complete!");
  console.log(`   Categories: ${categoryEmbeddings.length}`);
  console.log(
    `   Embedding dimensions: ${categoryEmbeddings[0].embedding.length}`,
  );
  console.log(`   Merchant overrides: ${Object.keys(merchantLookup).length}`);
  console.log("\nFiles saved to ./embeddings/");
}

train().catch(console.error);
