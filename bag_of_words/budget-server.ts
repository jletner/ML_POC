import * as tf from "@tensorflow/tfjs";
import * as fs from "node:fs";
import * as http from "node:http";
import * as path from "node:path";
import { execSync } from "node:child_process";

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

  // First check exact merchant match from corrections
  const exactMatch = merchantLookup.get(merchant.toLowerCase());
  if (exactMatch) {
    // Return exact match with high confidence, but still show ML alternatives
    const vec = [...merchantToVector(merchant), normalizeAmount(amount)];
    const pred = model.predict(tf.tensor2d([vec])) as tf.Tensor;
    const probs = await pred.data();

    const allRanked = categories
      .map((cat, i) => ({ category: cat, confidence: probs[i] }))
      .sort((a, b) => b.confidence - a.confidence);

    // Add user-defined categories with 0 confidence
    const userDefined = Array.from(userDefinedCategories)
      .filter((cat) => !categories.includes(cat))
      .map((cat) => ({ category: cat, confidence: 0 }));

    return {
      merchant,
      amount,
      prediction: exactMatch,
      confidence: 1.0,
      matchType: "exact",
      alternatives: [...allRanked, ...userDefined],
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

  return {
    merchant,
    amount,
    prediction: best.category,
    confidence: best.confidence,
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

async function retrainModel(): Promise<{ success: boolean; output: string }> {
  console.log("Starting model retraining...");
  try {
    const output = execSync("npx tsx train-budget.ts", {
      encoding: "utf-8",
      cwd: process.cwd(),
      timeout: 120000, // 2 minute timeout
    });
    console.log("Retraining complete, reloading model and lookup...");
    await loadModel();
    userDefinedCategories.clear(); // Clear since they're now in the model
    merchantLookup = loadMerchantLookup();
    return { success: true, output };
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
