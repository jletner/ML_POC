import * as path from "node:path";
import * as dotenv from "dotenv";
dotenv.config({ path: path.join(import.meta.dirname, ".env") });

import OpenAI from "openai";
import * as fs from "node:fs";
import * as http from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { execSync } from "node:child_process";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface CategoryEmbedding {
  category: string;
  embedding: number[];
  merchantExamples: string[];
}

let categoryEmbeddings: CategoryEmbedding[] = [];
let merchantLookup: Record<string, string> = {};
let userDefinedCategories: Set<string> = new Set(); // Track categories added via UI
let merchantEmbeddingCache: Record<string, number[]> = {}; // Cache to avoid repeated API calls
const SCRIPT_DIR = import.meta.dirname;
const CACHE_FILE = path.join(
  SCRIPT_DIR,
  "embeddings/merchant-embedding-cache.json",
);
const CATEGORY_EMBEDDINGS_FILE = path.join(
  SCRIPT_DIR,
  "embeddings/category-embeddings.json",
);
const MERCHANT_LOOKUP_FILE = path.join(
  SCRIPT_DIR,
  "embeddings/merchant-lookup.json",
);
const CORRECTIONS_FILE = path.join(SCRIPT_DIR, "data/user-corrections.json");

// Load embeddings
function loadEmbeddings() {
  categoryEmbeddings = JSON.parse(
    fs.readFileSync(CATEGORY_EMBEDDINGS_FILE, "utf-8"),
  );
  merchantLookup = JSON.parse(fs.readFileSync(MERCHANT_LOOKUP_FILE, "utf-8"));

  // Load merchant embedding cache
  try {
    merchantEmbeddingCache = JSON.parse(fs.readFileSync(CACHE_FILE, "utf-8"));
    console.log(
      `Loaded ${Object.keys(merchantEmbeddingCache).length} cached merchant embeddings`,
    );
  } catch {
    merchantEmbeddingCache = {};
  }

  // Load user-defined categories from corrections
  try {
    const corrections = JSON.parse(fs.readFileSync(CORRECTIONS_FILE, "utf-8"));
    const embeddedCats = new Set(categoryEmbeddings.map((c) => c.category));
    for (const c of corrections) {
      if (!embeddedCats.has(c.category)) {
        userDefinedCategories.add(c.category);
      }
    }
  } catch {
    // No corrections file yet
  }

  console.log(`Loaded ${categoryEmbeddings.length} category embeddings`);
  console.log(
    `Loaded ${Object.keys(merchantLookup).length} merchant overrides`,
  );
  if (userDefinedCategories.size > 0) {
    console.log(
      `User-defined categories (pending retrain): ${Array.from(userDefinedCategories).join(", ")}`,
    );
  }
}

// Get all categories including user-defined ones
function getAllCategories(): { category: string; confidence: number }[] {
  const embedded = categoryEmbeddings.map((c) => ({
    category: c.category,
    confidence: 0,
  }));
  const userDefined = Array.from(userDefinedCategories).map((cat) => ({
    category: cat,
    confidence: 0,
  }));
  return [...embedded, ...userDefined];
}

// Get embedding from OpenAI (with caching)
async function getEmbedding(
  text: string,
): Promise<{ embedding: number[]; cacheHit: boolean }> {
  const cacheKey = text.toLowerCase().trim();

  // Check cache first
  if (merchantEmbeddingCache[cacheKey]) {
    console.log(`Cache hit for: ${text}`);
    return { embedding: merchantEmbeddingCache[cacheKey], cacheHit: true };
  }

  // Call OpenAI API
  console.log(`Cache miss, calling OpenAI for: ${text}`);
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    dimensions: 512,
  });
  const embedding = response.data[0].embedding;

  // Save to cache
  merchantEmbeddingCache[cacheKey] = embedding;
  saveMerchantCache();

  return { embedding, cacheHit: false };
}

// Save merchant embedding cache to disk
function saveMerchantCache() {
  fs.writeFileSync(CACHE_FILE, JSON.stringify(merchantEmbeddingCache, null, 2));
}

// Cosine similarity between two vectors
function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Classify a transaction using vector similarity
async function classify(merchant: string, amount: number) {
  console.log("\n" + "=".repeat(60));
  console.log(`[VECTOR CLASSIFY] Input: "${merchant}" $${amount}`);

  // First check exact merchant match from corrections
  const exactMatch = merchantLookup[merchant.toLowerCase()];
  if (exactMatch) {
    console.log(
      `[VECTOR CLASSIFY] EXACT MATCH found in corrections → "${exactMatch}"`,
    );
    console.log(
      `[VECTOR CLASSIFY] Reason: User previously corrected this merchant`,
    );

    // Get embedding anyway to show alternatives
    const { embedding, cacheHit } = await getEmbedding(merchant);
    console.log(
      `[VECTOR CLASSIFY] Embedding source: ${cacheHit ? "CACHE" : "OpenAI API"}`,
    );

    const similarities = categoryEmbeddings.map((cat) => ({
      category: cat.category,
      confidence: (cosineSimilarity(embedding, cat.embedding) + 1) / 2, // Normalize to 0-1
    }));
    // Add user-defined categories (no embedding yet, so 0 confidence)
    for (const cat of userDefinedCategories) {
      similarities.push({ category: cat, confidence: 0 });
    }
    similarities.sort((a, b) => b.confidence - a.confidence);

    console.log(
      `[VECTOR CLASSIFY] (Vector similarity would have predicted: "${similarities[0].category}" at ${(similarities[0].confidence * 100).toFixed(1)}%)`,
    );
    console.log("=".repeat(60));

    // Override the exact match category to show 100% confidence in alternatives
    const allAlternatives = similarities.map((alt) =>
      alt.category === exactMatch ? { ...alt, confidence: 1.0 } : alt,
    );
    allAlternatives.sort((a, b) => b.confidence - a.confidence);

    // Build detailed reasoning for exact match
    const exactReasoning = {
      summary: `Exact match from user correction database.`,
      matchType: "exact",
      model: "OpenAI text-embedding-3-small",
      details: {
        source: "user-corrections.json (merchant-lookup.json)",
        merchantLookup: merchant.toLowerCase(),
        correctedTo: exactMatch,
        bypassedVectorSearch: false, // Still computed for comparison
      },
      embeddingDetails: {
        source: cacheHit ? "cache (instant)" : "OpenAI API (network call)",
        dimensions: embedding.length,
        note: "Embedding still computed to show vector-based alternatives",
      },
      vectorComparison: {
        note: "Vector similarity was computed for comparison purposes",
        vectorWouldHavePredicted: similarities[0].category,
        vectorConfidence: (similarities[0].confidence * 100).toFixed(1) + "%",
        matchesCorrection: similarities[0].category === exactMatch,
      },
      inference: {
        totalCategories: categoryEmbeddings.length,
        estimatedLatency: cacheHit ? "~1ms (cached)" : "~200-500ms (API call)",
      },
    };

    return {
      merchant,
      amount,
      prediction: exactMatch,
      confidence: 1.0,
      matchType: "exact",
      embeddingSource: cacheHit ? "cache" : "openai",
      reasoning: exactReasoning,
      alternatives: allAlternatives,
    };
  }

  // Get embedding for merchant
  const { embedding, cacheHit } = await getEmbedding(merchant);
  console.log(
    `[VECTOR CLASSIFY] Embedding source: ${cacheHit ? "CACHE (instant)" : "OpenAI API (network call)"}`,
  );
  console.log(`[VECTOR CLASSIFY] Embedding dimensions: ${embedding.length}`);

  // Show merchant embedding vector (beginning, middle, end for a complete picture)
  const sampleDims = 8;
  const startSlice = embedding.slice(0, sampleDims).map((v) => v.toFixed(4));
  const midStart =
    Math.floor(embedding.length / 2) - Math.floor(sampleDims / 2);
  const midSlice = embedding
    .slice(midStart, midStart + sampleDims)
    .map((v) => v.toFixed(4));
  const endSlice = embedding.slice(-sampleDims).map((v) => v.toFixed(4));

  console.log(`[VECTOR CLASSIFY] Merchant embedding for "${merchant}":`);
  console.log(
    `[VECTOR CLASSIFY]   Start [0-${sampleDims - 1}]:    [${startSlice.join(", ")}]`,
  );
  console.log(
    `[VECTOR CLASSIFY]   Middle [${midStart}-${midStart + sampleDims - 1}]: [${midSlice.join(", ")}]`,
  );
  console.log(
    `[VECTOR CLASSIFY]   End [${embedding.length - sampleDims}-${embedding.length - 1}]:  [${endSlice.join(", ")}]`,
  );

  // Calculate stats
  const min = Math.min(...embedding).toFixed(4);
  const max = Math.max(...embedding).toFixed(4);
  const avg = (embedding.reduce((a, b) => a + b, 0) / embedding.length).toFixed(
    4,
  );
  console.log(`[VECTOR CLASSIFY]   Stats: min=${min}, max=${max}, avg=${avg}`);

  // Calculate similarity to each category
  const similarities: {
    category: string;
    confidence: number;
    rawSimilarity?: number;
    examples?: string[];
    categoryVector?: number[];
  }[] = categoryEmbeddings.map((cat) => {
    const rawSim = cosineSimilarity(embedding, cat.embedding);
    return {
      category: cat.category,
      confidence: (rawSim + 1) / 2, // Normalize to 0-1
      rawSimilarity: rawSim,
      examples: cat.merchantExamples,
      categoryVector: cat.embedding.slice(0, sampleDims),
    };
  });

  // Add user-defined categories (no embedding yet, so 0 confidence)
  for (const cat of userDefinedCategories) {
    similarities.push({ category: cat, confidence: 0 });
  }

  // Sort by similarity
  similarities.sort((a, b) => b.confidence - a.confidence);

  const best = similarities[0];
  const top3 = similarities.slice(0, 3);

  // Build detailed reasoning
  const bestExamples = best.examples?.slice(0, 3) || [];
  const secondBest = top3.length > 1 ? top3[1] : null;

  const reasoning = {
    summary: `"${merchant}" is semantically similar to ${best.category} merchants (e.g., ${bestExamples.join(", ") || "none"}).`,
    matchType: "vector",
    model: "OpenAI text-embedding-3-small",
    embeddingDetails: {
      source: cacheHit ? "cache (instant)" : "OpenAI API (network call)",
      dimensions: embedding.length,
      vectorSample: {
        first3: embedding.slice(0, 3).map((v) => v.toFixed(4)),
        last3: embedding.slice(-3).map((v) => v.toFixed(4)),
      },
      vectorStats: {
        min: Math.min(...embedding).toFixed(4),
        max: Math.max(...embedding).toFixed(4),
        mean: (embedding.reduce((a, b) => a + b, 0) / embedding.length).toFixed(
          4,
        ),
      },
    },
    similarityAnalysis: {
      method: "Cosine Similarity",
      rawCosineSimilarity: best.rawSimilarity?.toFixed(4),
      normalizedConfidence: (best.confidence * 100).toFixed(2) + "%",
      normalizationFormula: "(cosine + 1) / 2 to map [-1,1] → [0,1]",
    },
    categoryMatch: {
      matchedCategory: best.category,
      similarMerchants: bestExamples,
      categoryHasSamples: (best.examples?.length || 0) > 0,
    },
    confidenceBreakdown: {
      top3: top3.map((cat, i) => ({
        rank: i + 1,
        category: cat.category,
        confidence: (cat.confidence * 100).toFixed(2) + "%",
        cosineSimilarity: cat.rawSimilarity?.toFixed(4),
        exampleMerchants: cat.examples?.slice(0, 2) || [],
      })),
      marginOverSecond: secondBest
        ? ((best.confidence - secondBest.confidence) * 100).toFixed(2) + "%"
        : "N/A",
      interpretation:
        best.confidence > 0.85
          ? "High confidence - strong semantic match"
          : best.confidence > 0.7
            ? "Moderate confidence - reasonable semantic match"
            : "Low confidence - weak semantic match, may need review",
    },
    inference: {
      totalCategories: categoryEmbeddings.length,
      estimatedLatency: cacheHit ? "~1ms (cached)" : "~200-500ms (API call)",
    },
  };

  console.log(`[VECTOR CLASSIFY] Cosine similarities to category embeddings:`);
  top3.forEach((cat, i) => {
    const exampleList = cat.examples?.slice(0, 2).join(", ") || "N/A";
    const catVectorStr = cat.categoryVector
      ? cat.categoryVector.map((v) => v.toFixed(4)).join(", ")
      : "N/A";
    console.log(
      `[VECTOR CLASSIFY]   ${i + 1}. ${cat.category}: ${(cat.confidence * 100).toFixed(2)}% (cosine: ${cat.rawSimilarity?.toFixed(4)})`,
    );
    console.log(
      `[VECTOR CLASSIFY]      Category vector (first ${sampleDims} dims): [${catVectorStr}...]`,
    );
    console.log(`[VECTOR CLASSIFY]      Examples: ${exampleList}`);
  });
  console.log(
    `[VECTOR CLASSIFY] Decision: "${best.category}" (${(best.confidence * 100).toFixed(1)}%)`,
  );
  console.log(
    `[VECTOR CLASSIFY] Reasoning: ${JSON.stringify(reasoning, null, 2)}`,
  );
  console.log("=".repeat(60));

  return {
    merchant,
    amount,
    prediction: best.category,
    confidence: best.confidence,
    matchType: "vector",
    embeddingSource: cacheHit ? "cache" : "openai",
    reasoning,
    topMatches: similarities.slice(0, 5),
    alternatives: similarities,
  };
}

// Handle user correction - save and reload
function handleCorrection(merchant: string, category: string) {
  // Add to user corrections file
  const correctionsPath = CORRECTIONS_FILE;
  let corrections: { merchant: string; amount: number; category: string }[] =
    [];

  try {
    corrections = JSON.parse(fs.readFileSync(correctionsPath, "utf-8"));
  } catch {
    corrections = [];
  }

  // Add or update correction
  const existing = corrections.find(
    (c) => c.merchant.toLowerCase() === merchant.toLowerCase(),
  );
  if (existing) {
    existing.category = category;
  } else {
    corrections.push({ merchant, amount: 0, category });
  }

  fs.writeFileSync(correctionsPath, JSON.stringify(corrections, null, 2));

  // Update in-memory lookup
  merchantLookup[merchant.toLowerCase()] = category;

  return {
    success: true,
    message: `Saved correction: ${merchant} → ${category}`,
  };
}

function saveFeedback(merchant: string, amount: number, category: string) {
  const correctionsPath = CORRECTIONS_FILE;
  let corrections: {
    merchant: string;
    amount: number;
    category: string;
    timestamp?: string;
  }[] = [];

  try {
    corrections = JSON.parse(fs.readFileSync(correctionsPath, "utf-8"));
  } catch {
    corrections = [];
  }

  const isNewCategory =
    !categoryEmbeddings.some((c) => c.category === category) &&
    !userDefinedCategories.has(category);

  // Track new user-defined category immediately
  if (isNewCategory) {
    userDefinedCategories.add(category);
    console.log(`Added new user-defined category: ${category}`);
  }

  corrections.push({
    merchant,
    amount,
    category,
    timestamp: new Date().toISOString(),
  });
  fs.writeFileSync(correctionsPath, JSON.stringify(corrections, null, 2));

  // Update in-memory lookup
  merchantLookup[merchant.toLowerCase()] = category;

  return {
    success: true,
    totalCorrections: corrections.length,
    newCategory: isNewCategory,
  };
}

async function retrainModel(): Promise<{ success: boolean; output: string }> {
  console.log("Starting embedding retraining...");
  try {
    const output = execSync("npx tsx --env-file=.env train-embeddings.ts", {
      encoding: "utf-8",
      cwd: process.cwd(),
      timeout: 120000,
    });
    console.log("Retraining complete, reloading embeddings...");
    userDefinedCategories.clear(); // Clear since they'll now have embeddings
    loadEmbeddings();
    return { success: true, output };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("Retraining failed:", message);
    return { success: false, output: message };
  }
}

function parseBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

const HTML_PAGE = `<!DOCTYPE html>
<html>
<head>
  <title>Vector Budget Classifier</title>
  <style>
    * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    body { max-width: 600px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
    h1 { color: #333; }
    .badge { display: inline-block; background: #6f42c1; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    label { display: block; margin-bottom: 5px; font-weight: 500; color: #555; }
    input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 15px; font-size: 16px; }
    button { background: #6f42c1; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; }
    button:hover { background: #5a32a3; }
    button.secondary { background: #28a745; }
    button.secondary:hover { background: #1e7e34; }
    button.retrain { background: #007bff; margin-top: 20px; width: 100%; }
    button.retrain:hover { background: #0056b3; }
    button.retrain:disabled { background: #ccc; cursor: not-allowed; }
    .retrain-status { margin-top: 10px; padding: 10px; border-radius: 4px; display: none; }
    .retrain-status.loading { display: block; background: #fff3cd; color: #856404; }
    .retrain-status.success { display: block; background: #d4edda; color: #155724; }
    .retrain-status.error { display: block; background: #f8d7da; color: #721c24; }
    #result { display: none; }
    .prediction { font-size: 24px; color: #6f42c1; margin: 10px 0; }
    .confidence { color: #666; }
    .match-type { font-size: 12px; padding: 2px 8px; border-radius: 4px; margin-left: 10px; }
    .match-type.exact { background: #28a745; color: white; }
    .match-type.vector { background: #6f42c1; color: white; }
    .embedding-source { font-size: 12px; padding: 2px 8px; border-radius: 4px; margin-left: 8px; }
    .embedding-source.cache { background: #17a2b8; color: white; }
    .embedding-source.openai { background: #fd7e14; color: white; }
    .alternatives { margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; }
    .alt-item { display: flex; justify-content: space-between; padding: 5px 0; color: #888; }
    select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 15px; font-size: 16px; }
    .feedback-section { display: none; margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; }
    .success { color: #28a745; font-weight: 500; }
    .custom-input { display: none; }
    .custom-input.visible { display: block; }
  </style>
</head>
<body>
  <h1>Vector Budget Classifier <span class="badge">Embeddings</span></h1>
  
  <div class="card" style="background: #f3e8ff; border-left: 4px solid #6f42c1;">
    <h3 style="margin-top: 0; color: #5a32a3;">How This Model Works</h3>
    <p style="margin: 0; color: #333; line-height: 1.6;">
      <strong>Semantic Embeddings:</strong> This classifier uses OpenAI's text-embedding-3-small model to convert 
      merchant names into 512-dimensional vectors that capture semantic meaning. Each category is represented 
      by an average embedding of its training merchants.<br><br>
      <strong>Suggestions:</strong> Predictions are based on cosine similarity between the merchant embedding and 
      category embeddings. This approach understands meaning, not just words — so "JavaHut Espresso" matches 
      "Coffee" even without the word "coffee" appearing in training. Higher confidence = closer semantic match.
    </p>
  </div>
  
  <div class="card">
    <form id="classifyForm">
      <label>Merchant Name</label>
      <input type="text" id="merchant" placeholder="e.g. Starbucks, JavaHut Espresso" required>
      <label>Amount ($)</label>
      <input type="number" id="amount" step="0.01" placeholder="e.g. 5.50" required>
      <button type="submit">Classify</button>
    </form>
  </div>
  
  <div class="card">
    <button class="retrain" id="retrainBtn" onclick="retrainModel()">Retrain Embeddings</button>
    <div class="retrain-status" id="retrainStatus"></div>
  </div>
  
  <div class="card" id="result">
    <h3>Classification Result</h3>
    <div>
      <span class="prediction" id="prediction"></span>
      <span class="match-type" id="matchType"></span>
      <span class="embedding-source" id="embeddingSource"></span>
    </div>
    <div class="confidence" id="confidence"></div>
    <div class="alternatives" id="alternatives"></div>
    
    <div class="feedback-section" id="feedbackSection">
      <label>Wrong? Select correct category:</label>
      <select id="correctCategory" onchange="toggleCustomInput()"></select>
      <div class="custom-input" id="customInputDiv">
        <label>Enter new category name:</label>
        <input type="text" id="customCategory" placeholder="e.g. Electronics">
      </div>
      <button class="secondary" onclick="submitFeedback()">Submit Correction</button>
      <div id="feedbackResult"></div>
    </div>
  </div>
  
  <script>
    let lastMerchant = "";
    let lastAmount = 0;
    
    document.getElementById("classifyForm").onsubmit = async (e) => {
      e.preventDefault();
      const merchant = document.getElementById("merchant").value;
      const amount = parseFloat(document.getElementById("amount").value);
      lastMerchant = merchant;
      lastAmount = amount;
      
      const res = await fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ merchant, amount })
      });
      const data = await res.json();
      
      document.getElementById("result").style.display = "block";
      document.getElementById("prediction").textContent = data.prediction;
      
      const matchType = document.getElementById("matchType");
      matchType.textContent = data.matchType;
      matchType.className = "match-type " + data.matchType;
      
      const embeddingSource = document.getElementById("embeddingSource");
      embeddingSource.textContent = data.embeddingSource === "cache" ? "From Cache" : "From OpenAI";
      embeddingSource.className = "embedding-source " + data.embeddingSource;
      
      document.getElementById("confidence").textContent = "Confidence: " + (data.confidence * 100).toFixed(1) + "%";
      
      let altHtml = "<strong>Other possibilities:</strong>";
      data.alternatives.slice(0, 5).forEach(alt => {
        altHtml += "<div class='alt-item'><span>" + alt.category + "</span><span>" + (alt.confidence * 100).toFixed(1) + "%</span></div>";
      });
      document.getElementById("alternatives").innerHTML = altHtml;
      
      const select = document.getElementById("correctCategory");
      select.innerHTML = data.alternatives.map(a => "<option value='" + a.category + "'>" + a.category + "</option>").join("") + "<option value='__custom__'>+ Add custom category...</option>";
      document.getElementById("feedbackSection").style.display = "block";
      document.getElementById("feedbackResult").textContent = "";
      document.getElementById("customInputDiv").classList.remove("visible");
      document.getElementById("customCategory").value = "";
    };
    
    function toggleCustomInput() {
      const select = document.getElementById("correctCategory");
      const customDiv = document.getElementById("customInputDiv");
      if (select.value === "__custom__") {
        customDiv.classList.add("visible");
      } else {
        customDiv.classList.remove("visible");
      }
    }
    
    async function submitFeedback() {
      let category = document.getElementById("correctCategory").value;
      if (category === "__custom__") {
        category = document.getElementById("customCategory").value.trim();
        if (!category) {
          alert("Please enter a custom category name");
          return;
        }
      }
      const res = await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ merchant: lastMerchant, amount: lastAmount, category })
      });
      const data = await res.json();
      document.getElementById("feedbackResult").innerHTML = "<span class='success'>Feedback saved! Total corrections: " + data.totalCorrections + (data.newCategory ? " (new category added!)" : "") + "</span>";
    }
    
    async function retrainModel() {
      const btn = document.getElementById("retrainBtn");
      const status = document.getElementById("retrainStatus");
      btn.disabled = true;
      btn.textContent = "Retraining...";
      status.className = "retrain-status loading";
      status.textContent = "Regenerating embeddings... This may take a minute.";
      
      try {
        const res = await fetch("/retrain", { method: "POST" });
        const data = await res.json();
        if (data.success) {
          status.className = "retrain-status success";
          status.textContent = "Embeddings retrained successfully! New predictions are now active.";
        } else {
          status.className = "retrain-status error";
          status.textContent = "Retraining failed: " + data.output;
        }
      } catch (err) {
        status.className = "retrain-status error";
        status.textContent = "Error: " + err.message;
      }
      btn.disabled = false;
      btn.textContent = "Retrain Embeddings";
    }
  </script>
</body>
</html>`;

// HTTP Server
const server = http.createServer(
  async (req: IncomingMessage, res: ServerResponse) => {
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");

    if (req.method === "OPTIONS") {
      res.writeHead(200);
      res.end();
      return;
    }

    const url = new URL(req.url || "/", `http://${req.headers.host}`);

    try {
      // Serve HTML UI
      if (
        req.method === "GET" &&
        (url.pathname === "/" || url.pathname === "/index.html")
      ) {
        res.setHeader("Content-Type", "text/html");
        res.writeHead(200);
        res.end(HTML_PAGE);
        return;
      }

      res.setHeader("Content-Type", "application/json");

      // POST /classify - from UI form
      if (url.pathname === "/classify" && req.method === "POST") {
        const body = await parseBody(req);
        const { merchant, amount } = JSON.parse(body);
        const result = await classify(merchant, amount);
        res.writeHead(200);
        res.end(JSON.stringify(result));
        return;
      }

      // GET /classify?merchant=xxx&amount=xxx
      if (url.pathname === "/classify" && req.method === "GET") {
        const merchant = url.searchParams.get("merchant");
        const amount = parseFloat(url.searchParams.get("amount") || "0");

        if (!merchant) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: "merchant parameter required" }));
          return;
        }

        const result = await classify(merchant, amount);
        res.writeHead(200);
        res.end(JSON.stringify(result, null, 2));
        return;
      }

      // POST /correct - body: {merchant, category}
      if (url.pathname === "/correct" && req.method === "POST") {
        let body = "";
        for await (const chunk of req) {
          body += chunk;
        }
        const { merchant, category } = JSON.parse(body);

        if (!merchant || !category) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: "merchant and category required" }));
          return;
        }

        const result = handleCorrection(merchant, category);
        res.writeHead(200);
        res.end(JSON.stringify(result));
        return;
      }

      // POST /feedback - from UI form
      if (url.pathname === "/feedback" && req.method === "POST") {
        const body = await parseBody(req);
        const { merchant, amount, category } = JSON.parse(body);
        const result = saveFeedback(merchant, amount, category);
        res.writeHead(200);
        res.end(JSON.stringify(result));
        return;
      }

      // POST /retrain - trigger embedding regeneration
      if (url.pathname === "/retrain" && req.method === "POST") {
        const result = await retrainModel();
        res.writeHead(200);
        res.end(JSON.stringify(result));
        return;
      }

      // GET /categories
      if (url.pathname === "/categories" && req.method === "GET") {
        res.writeHead(200);
        res.end(
          JSON.stringify({
            categories: [
              ...categoryEmbeddings.map((c) => c.category),
              ...Array.from(userDefinedCategories),
            ],
          }),
        );
        return;
      }

      // GET /health
      if (url.pathname === "/health") {
        res.writeHead(200);
        res.end(
          JSON.stringify({
            status: "ok",
            model: "text-embedding-3-small",
            categories: categoryEmbeddings.length,
          }),
        );
        return;
      }

      res.writeHead(404);
      res.end(JSON.stringify({ error: "Not found" }));
    } catch (error) {
      console.error("Error:", error);
      res.writeHead(500);
      res.end(
        JSON.stringify({
          error: error instanceof Error ? error.message : "Unknown error",
        }),
      );
    }
  },
);

const PORT = 3002;

loadEmbeddings();
server.listen(PORT, () => {
  console.log(`\n🚀 Vector Budget Server running on http://localhost:${PORT}`);
  console.log("\nOpen http://localhost:3002 in your browser for the UI");
  console.log("\nAPI Endpoints:");
  console.log(`  GET  /classify?merchant=xxx&amount=xxx`);
  console.log(`  POST /classify  {merchant, amount}`);
  console.log(`  POST /feedback  {merchant, amount, category}`);
  console.log(`  POST /retrain`);
  console.log(`  GET  /categories`);
  console.log(`  GET  /health`);
});
