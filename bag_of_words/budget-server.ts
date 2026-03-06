import * as tf from "@tensorflow/tfjs";
import * as fs from "node:fs";
import * as http from "node:http";
import * as path from "node:path";
import { execSync } from "node:child_process";
import * as dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config({ path: path.join(import.meta.dirname, ".env") });

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const SCRIPT_DIR = import.meta.dirname;
const MODEL_DIR = path.join(SCRIPT_DIR, "budget-model");
const DATA_DIR = path.join(SCRIPT_DIR, "data");

let model: tf.LayersModel | null = null;
let vocab: Map<string, number> = new Map();
let categories: string[] = [];
let userDefinedCategories: Set<string> = new Set(); // Track categories added via UI

// Get all categories including user-defined ones
function getAllCategories(): string[] {
  return [...categories, ...Array.from(userDefinedCategories)];
}

async function loadModel() {
  const modelJSON = JSON.parse(
    fs.readFileSync(path.join(MODEL_DIR, "model.json"), "utf-8"),
  );
  const weightsBuffer = fs.readFileSync(path.join(MODEL_DIR, "weights.bin"));
  const vocabObj = JSON.parse(
    fs.readFileSync(path.join(MODEL_DIR, "vocab.json"), "utf-8"),
  );
  categories = JSON.parse(
    fs.readFileSync(path.join(MODEL_DIR, "categories.json"), "utf-8"),
  );

  vocab = new Map(Object.entries(vocabObj).map(([k, v]) => [k, v as number]));

  // Log weights.bin details
  console.log("\n" + "=".repeat(60));
  console.log("[WEIGHTS.BIN] Loading neural network weights...");
  console.log(
    `[WEIGHTS.BIN] File size: ${weightsBuffer.length.toLocaleString()} bytes (${(weightsBuffer.length / 1024).toFixed(2)} KB)`,
  );

  const weights = new Float32Array(
    weightsBuffer.buffer,
    weightsBuffer.byteOffset,
    weightsBuffer.length / 4,
  );
  console.log(
    `[WEIGHTS.BIN] Total parameters: ${weights.length.toLocaleString()}`,
  );

  // Parse weight manifest for layer details
  const weightManifest = modelJSON.weightsManifest[0].weights;
  console.log(`[WEIGHTS.BIN] Weight tensors:`);
  let offset = 0;
  for (const w of weightManifest) {
    const size = w.shape.reduce((a: number, b: number) => a * b, 1);
    const layerWeights = weights.slice(offset, offset + size);
    const min = Math.min(...layerWeights).toFixed(4);
    const max = Math.max(...layerWeights).toFixed(4);
    const mean = (
      layerWeights.reduce((a, b) => a + b, 0) / layerWeights.length
    ).toFixed(4);
    console.log(
      `[WEIGHTS.BIN]   ${w.name}: shape=${JSON.stringify(w.shape)}, params=${size.toLocaleString()}, range=[${min}, ${max}], mean=${mean}`,
    );
    offset += size;
  }

  // Overall stats
  const overallMin = Math.min(...weights).toFixed(4);
  const overallMax = Math.max(...weights).toFixed(4);
  const overallMean = (
    weights.reduce((a, b) => a + b, 0) / weights.length
  ).toFixed(4);
  console.log(
    `[WEIGHTS.BIN] Overall stats: min=${overallMin}, max=${overallMax}, mean=${overallMean}`,
  );
  console.log("=".repeat(60) + "\n");

  model = await tf.loadLayersModel(
    tf.io.fromMemory(
      modelJSON.modelTopology,
      modelJSON.weightsManifest[0].weights,
      new Uint8Array(weightsBuffer).buffer,
    ),
  );
  console.log("Budget model loaded successfully!");
}

function merchantToVector(merchant: string): number[] {
  const vector = new Array(vocab.size).fill(0);
  const words = merchant.toLowerCase().split(/\s+/);
  for (const word of words) {
    const idx = vocab.get(word) ?? 0;
    vector[idx] = 1;
  }
  return vector;
}

function normalizeAmount(amount: number): number {
  return Math.log10(amount + 1) / 4;
}

// Load user corrections for exact matching
function loadMerchantLookup(): Map<string, string> {
  try {
    const corrections = JSON.parse(
      fs.readFileSync(path.join(DATA_DIR, "user-corrections.json"), "utf-8"),
    );
    const lookup = new Map<string, string>();
    // Use most recent correction for each merchant
    // Also populate userDefinedCategories with any new categories from corrections
    for (const c of corrections) {
      lookup.set(c.merchant.toLowerCase(), c.category);
      if (!categories.includes(c.category)) {
        userDefinedCategories.add(c.category);
      }
    }
    return lookup;
  } catch {
    return new Map();
  }
}

let merchantLookup = loadMerchantLookup();

async function classify(merchant: string, amount: number) {
  if (!model) throw new Error("Model not loaded");

  // Extract words and identify which are in vocabulary
  const words = merchant.toLowerCase().split(/\s+/);
  const knownWords = words.filter((w) => vocab.has(w) && vocab.get(w) !== 0);
  const unknownWords = words.filter((w) => !vocab.has(w) || vocab.get(w) === 0);

  console.log("\n" + "=".repeat(60));
  console.log(`[BOW CLASSIFY] Input: "${merchant}" $${amount}`);
  console.log(`[BOW CLASSIFY] Words parsed: [${words.join(", ")}]`);
  console.log(
    `[BOW CLASSIFY] Known vocabulary words: [${knownWords.join(", ")}]`,
  );
  if (unknownWords.length > 0) {
    console.log(
      `[BOW CLASSIFY] Unknown words (mapped to <UNK>): [${unknownWords.join(", ")}]`,
    );
  }
  console.log(
    `[BOW CLASSIFY] Normalized amount: ${normalizeAmount(amount).toFixed(4)} (from $${amount})`,
  );

  // First check exact merchant match from corrections
  const exactMatch = merchantLookup.get(merchant.toLowerCase());
  if (exactMatch) {
    console.log(
      `[BOW CLASSIFY] EXACT MATCH found in corrections → "${exactMatch}"`,
    );
    console.log(
      `[BOW CLASSIFY] Reason: User previously corrected this merchant`,
    );

    // Return exact match with high confidence, but still show ML alternatives
    const vec = [...merchantToVector(merchant), normalizeAmount(amount)];
    const pred = model.predict(tf.tensor2d([vec])) as tf.Tensor;
    const probs = await pred.data();

    const allRanked = categories
      .map((cat, i) => ({ category: cat, confidence: probs[i] }))
      .sort((a, b) => b.confidence - a.confidence);

    console.log(
      `[BOW CLASSIFY] (ML would have predicted: "${allRanked[0].category}" at ${(allRanked[0].confidence * 100).toFixed(1)}%)`,
    );
    console.log("=".repeat(60));

    // Add user-defined categories with 0 confidence
    const userDefined = Array.from(userDefinedCategories)
      .filter((cat) => !categories.includes(cat))
      .map((cat) => ({ category: cat, confidence: 0 }));

    // Override the exact match category to show 100% confidence in alternatives
    const allAlternatives = [...allRanked, ...userDefined].map((alt) =>
      alt.category === exactMatch ? { ...alt, confidence: 1.0 } : alt,
    );
    allAlternatives.sort((a, b) => b.confidence - a.confidence);

    // Build detailed reasoning for exact match
    const exactReasoning = {
      summary: `Exact match from user correction database.`,
      matchType: "exact",
      model: "Bag of Words (TensorFlow.js)",
      details: {
        source: "user-corrections.json",
        merchantLookup: merchant.toLowerCase(),
        correctedTo: exactMatch,
        bypassedML: true,
      },
      vocabularyAnalysis: {
        inputWords: words,
        knownWords: knownWords,
        unknownWords: unknownWords,
        vocabularySize: vocab.size,
        coveragePercent:
          ((knownWords.length / words.length) * 100).toFixed(1) + "%",
      },
      mlComparison: {
        note: "ML prediction was bypassed due to exact match",
        mlWouldHavePredicted: allRanked[0].category,
        mlConfidence: (allRanked[0].confidence * 100).toFixed(1) + "%",
      },
    };

    return {
      merchant,
      amount,
      prediction: exactMatch,
      confidence: 1.0,
      matchType: "exact",
      reasoning: exactReasoning,
      alternatives: allAlternatives,
    };
  }

  const vec = [...merchantToVector(merchant), normalizeAmount(amount)];
  const pred = model.predict(tf.tensor2d([vec])) as tf.Tensor;
  const probs = await pred.data();

  // Get all predictions sorted by confidence
  const allRanked = categories
    .map((cat, i) => ({ category: cat, confidence: probs[i] }))
    .sort((a, b) => b.confidence - a.confidence);

  // Add user-defined categories with 0 confidence
  const userDefined = Array.from(userDefinedCategories)
    .filter((cat) => !categories.includes(cat))
    .map((cat) => ({ category: cat, confidence: 0 }));

  // Use the top ranked item for consistency (same source for confidence)
  const best = allRanked[0];
  const top3 = allRanked.slice(0, 3);

  // Build detailed reasoning explanation
  const reasoning = {
    summary:
      knownWords.length === 0
        ? `No vocabulary matches found. Model relied on amount ($${amount}) and general patterns.`
        : `Word patterns [${knownWords.join(", ")}] strongly associated with "${best.category}" in training data.`,
    matchType: "model",
    model: "Bag of Words (TensorFlow.js)",
    architecture: {
      type: "Sequential Neural Network",
      layers:
        "Input(" +
        vocab.size +
        "+1) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Softmax(" +
        categories.length +
        ")",
      totalCategories: categories.length,
    },
    vocabularyAnalysis: {
      inputWords: words,
      knownWords: knownWords,
      unknownWords: unknownWords,
      vocabularySize: vocab.size,
      coveragePercent:
        ((knownWords.length / words.length) * 100).toFixed(1) + "%",
      note:
        unknownWords.length > 0
          ? `Unknown words mapped to <UNK> token (index 0)`
          : "All words recognized",
    },
    amountAnalysis: {
      rawAmount: amount,
      normalizedValue: normalizeAmount(amount).toFixed(4),
      normalizationMethod: "log10(amount + 1) / 4",
    },
    confidenceBreakdown: {
      top3: top3.map((cat, i) => ({
        rank: i + 1,
        category: cat.category,
        confidence: (cat.confidence * 100).toFixed(2) + "%",
        probability: cat.confidence.toFixed(4),
      })),
      marginOverSecond:
        top3.length > 1
          ? ((top3[0].confidence - top3[1].confidence) * 100).toFixed(2) + "%"
          : "N/A",
    },
    inference: {
      vectorSize: vocab.size + 1,
      activatedPositions: knownWords.length + (unknownWords.length > 0 ? 1 : 0),
      estimatedLatency: "~1ms (local inference)",
    },
  };

  console.log(`[BOW CLASSIFY] Neural network output probabilities:`);
  top3.forEach((cat, i) => {
    console.log(
      `[BOW CLASSIFY]   ${i + 1}. ${cat.category}: ${(cat.confidence * 100).toFixed(2)}%`,
    );
  });
  console.log(
    `[BOW CLASSIFY] Decision: "${best.category}" (${(best.confidence * 100).toFixed(1)}%)`,
  );
  console.log(`[BOW CLASSIFY] Reasoning: ${reasoning}`);
  console.log("=".repeat(60));

  return {
    merchant,
    amount,
    prediction: best.category,
    confidence: best.confidence,
    matchType: "model",
    reasoning,
    knownWords,
    unknownWords: unknownWords.length > 0 ? unknownWords : undefined,
    alternatives: [...allRanked, ...userDefined],
  };
}

function saveFeedback(merchant: string, amount: number, category: string) {
  const corrections = JSON.parse(
    fs.readFileSync(path.join(DATA_DIR, "user-corrections.json"), "utf-8"),
  );
  const isNewCategory = !categories.includes(category);
  corrections.push({
    merchant,
    amount,
    category,
    timestamp: new Date().toISOString(),
  });
  fs.writeFileSync(
    path.join(DATA_DIR, "user-corrections.json"),
    JSON.stringify(corrections, null, 2),
  );
  // Update lookup immediately so exact matches work before retrain
  merchantLookup.set(merchant.toLowerCase(), category);
  // Add new category to userDefinedCategories so it appears in dropdown immediately
  if (isNewCategory) {
    userDefinedCategories.add(category);
  }
  return {
    success: true,
    totalCorrections: corrections.length,
    newCategory: isNewCategory,
  };
}

function parseBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

// Generate synthetic merchant examples for a new category using OpenAI
async function generateSyntheticMerchants(
  category: string,
  count: number = 10,
): Promise<{ merchant: string; amount: number; category: string }[]> {
  console.log(
    `Generating ${count} synthetic merchants for new category: ${category}`,
  );

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            "You generate realistic merchant names for budget categorization training data. Return ONLY a JSON array of objects with 'merchant' (string) and 'amount' (number) fields. No markdown, no explanation.",
        },
        {
          role: "user",
          content: `Generate ${count} realistic merchant names for the category "${category}". Include variety in naming styles (e.g., "Joe's Auto Shop", "AutoZone", "Quick Lube Center"). Amounts should be realistic for this category. Return as JSON array.`,
        },
      ],
      temperature: 0.8,
    });

    const content = response.choices[0].message.content || "[]";
    // Parse JSON, handling potential markdown code blocks
    const jsonStr = content.replace(/```json\n?|```\n?/g, "").trim();
    const merchants = JSON.parse(jsonStr) as {
      merchant: string;
      amount: number;
    }[];

    console.log(
      `Generated ${merchants.length} synthetic merchants for ${category}`,
    );
    return merchants.map((m) => ({ ...m, category }));
  } catch (err: any) {
    console.error(`Failed to generate synthetic merchants: ${err.message}`);
    return [];
  }
}

// Check for new categories and generate synthetic data before retraining
async function generateSyntheticsForNewCategories(): Promise<string[]> {
  const corrections = JSON.parse(
    fs.readFileSync(path.join(DATA_DIR, "user-corrections.json"), "utf-8"),
  );
  const syntheticData = JSON.parse(
    fs.readFileSync(
      path.join(DATA_DIR, "synthetic-transactions.json"),
      "utf-8",
    ),
  );

  // Find categories in corrections that don't exist in synthetic data
  const syntheticCategories = new Set(
    syntheticData.map((t: any) => t.category),
  );
  const correctionCategories = new Set(corrections.map((c: any) => c.category));
  const newCategories = [...correctionCategories].filter(
    (c) => !syntheticCategories.has(c),
  );

  if (newCategories.length === 0) {
    return [];
  }

  console.log(
    `Found ${newCategories.length} new categories needing synthetic data: ${newCategories.join(", ")}`,
  );

  // Generate synthetic data for each new category
  for (const category of newCategories) {
    const synthetics = await generateSyntheticMerchants(category as string, 10);
    if (synthetics.length > 0) {
      syntheticData.push(...synthetics);
    }
  }

  // Save updated synthetic data
  fs.writeFileSync(
    path.join(DATA_DIR, "synthetic-transactions.json"),
    JSON.stringify(syntheticData, null, 2),
  );

  return newCategories as string[];
}

async function retrainModel(): Promise<{
  success: boolean;
  output: string;
  newCategories?: string[];
}> {
  console.log("Starting model retraining...");
  try {
    // Generate synthetic data for any new categories first
    const newCategories = await generateSyntheticsForNewCategories();
    if (newCategories.length > 0) {
      console.log(`Generated synthetic data for: ${newCategories.join(", ")}`);
    }

    const output = execSync("npx tsx train-budget.ts", {
      encoding: "utf-8",
      cwd: process.cwd(),
      timeout: 120000, // 2 minute timeout
    });
    console.log("Retraining complete, reloading model and lookup...");
    await loadModel();
    userDefinedCategories.clear(); // Clear since they're now in the model
    merchantLookup = loadMerchantLookup();
    return {
      success: true,
      output,
      newCategories: newCategories.length > 0 ? newCategories : undefined,
    };
  } catch (err: any) {
    console.error("Retraining failed:", err.message);
    return { success: false, output: err.message };
  }
}

const HTML_PAGE = `<!DOCTYPE html>
<html>
<head>
  <title>Budget Classifier</title>
  <style>
    * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    body { max-width: 600px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
    h1 { color: #333; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    label { display: block; margin-bottom: 5px; font-weight: 500; color: #555; }
    input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 15px; font-size: 16px; }
    button { background: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; }
    button:hover { background: #0056b3; }
    button.secondary { background: #28a745; }
    button.secondary:hover { background: #1e7e34; }
    button.retrain { background: #6f42c1; margin-top: 20px; width: 100%; }
    button.retrain:hover { background: #5a32a3; }
    button.retrain:disabled { background: #ccc; cursor: not-allowed; }
    .retrain-status { margin-top: 10px; padding: 10px; border-radius: 4px; display: none; }
    .retrain-status.loading { display: block; background: #fff3cd; color: #856404; }
    .retrain-status.success { display: block; background: #d4edda; color: #155724; }
    .retrain-status.error { display: block; background: #f8d7da; color: #721c24; }
    #result { display: none; }
    .prediction { font-size: 24px; color: #007bff; margin: 10px 0; }
    .confidence { color: #666; }
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
  <h1>Budget Classifier</h1>
  
  <div class="card" style="background: #e7f3ff; border-left: 4px solid #007bff;">
    <h3 style="margin-top: 0; color: #0056b3;">How This Model Works</h3>
    <p style="margin: 0; color: #333; line-height: 1.6;">
      <strong>Bag of Words + Neural Network:</strong> This classifier converts merchant names into binary vectors 
      where each position represents whether a specific word is present. A TensorFlow.js neural network then 
      learns patterns from labeled training data to predict categories.<br><br>
      <strong>Suggestions:</strong> The model outputs probability scores for each category based on word patterns it learned 
      during training. Higher confidence = stronger word pattern match. Works well for exact or similar merchant names 
      but may struggle with semantically similar merchants using different words (e.g., "Coffee House" vs "Espresso Bar").
    </p>
  </div>
  
  <div class="card">
    <form id="classifyForm">
      <label>Merchant Name</label>
      <input type="text" id="merchant" placeholder="e.g. Starbucks" required>
      <label>Amount ($)</label>
      <input type="number" id="amount" step="0.01" placeholder="e.g. 5.50" required>
      <button type="submit">Classify</button>
    </form>
  </div>
  
  <div class="card">
    <button class="retrain" id="retrainBtn" onclick="retrainModel()">Retrain Model</button>
    <div class="retrain-status" id="retrainStatus"></div>
  </div>
  
  <div class="card" id="result">
    <h3>Classification Result</h3>
    <div class="prediction" id="prediction"></div>
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
      document.getElementById("confidence").textContent = "Confidence: " + (data.confidence * 100).toFixed(1) + "%";
      
      let altHtml = "<strong>Other possibilities:</strong>";
      data.alternatives.slice(0, 3).forEach(alt => {
        altHtml += "<div class='alt-item'><span>" + alt.category + "</span><span>" + (alt.confidence * 100).toFixed(1) + "%</span></div>";
      });
      document.getElementById("alternatives").innerHTML = altHtml;
      
      // Populate feedback dropdown with ALL categories
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
      status.textContent = "Training model with corrections... This may take a minute.";
      
      try {
        const res = await fetch("/retrain", { method: "POST" });
        const data = await res.json();
        if (data.success) {
          status.className = "retrain-status success";
          status.textContent = "Model retrained successfully! New predictions are now active.";
        } else {
          status.className = "retrain-status error";
          status.textContent = "Retraining failed: " + data.output;
        }
      } catch (err) {
        status.className = "retrain-status error";
        status.textContent = "Error: " + err.message;
      }
      btn.disabled = false;
      btn.textContent = "Retrain Model";
    }
  </script>
</body>
</html>`;

const server = http.createServer(async (req, res) => {
  // CORS headers for cross-origin requests
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  // Handle preflight requests
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }

  try {
    if (
      req.method === "GET" &&
      (req.url === "/" || req.url === "/index.html")
    ) {
      res.setHeader("Content-Type", "text/html");
      res.end(HTML_PAGE);
      return;
    }

    res.setHeader("Content-Type", "application/json");

    if (req.method === "POST" && req.url === "/classify") {
      const body = await parseBody(req);
      const { merchant, amount } = JSON.parse(body);
      const result = await classify(merchant, amount);
      res.end(JSON.stringify(result));
    } else if (req.method === "POST" && req.url === "/feedback") {
      const body = await parseBody(req);
      const { merchant, amount, category } = JSON.parse(body);
      const result = saveFeedback(merchant, amount, category);
      res.end(JSON.stringify(result));
    } else if (req.method === "GET" && req.url === "/categories") {
      res.end(JSON.stringify({ categories }));
    } else if (req.method === "POST" && req.url === "/retrain") {
      const result = await retrainModel();
      res.end(JSON.stringify(result));
    } else if (req.method === "GET" && req.url === "/health") {
      res.end(JSON.stringify({ status: "ok", modelLoaded: model !== null }));
    } else {
      res.statusCode = 404;
      res.end(JSON.stringify({ error: "Not found" }));
    }
  } catch (err) {
    res.statusCode = 500;
    res.end(JSON.stringify({ error: String(err) }));
  }
});

const PORT = 3001;

loadModel()
  .then(() => {
    server.listen(PORT, () => {
      console.log(`Budget Classifier running at http://localhost:${PORT}`);
      console.log(`Open in browser to classify transactions`);
    });
  })
  .catch((err) => {
    console.error("Failed to load model:", err);
    process.exit(1);
  });
