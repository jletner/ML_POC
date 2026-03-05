import * as tf from "@tensorflow/tfjs";
import * as fs from "fs";

// Base categories (will be extended with user-defined ones)
const BASE_CATEGORIES = [
  "Coffee & Drinks",
  "Food & Dining",
  "Groceries",
  "Transportation",
  "Entertainment",
  "Shopping",
  "Bills & Utilities",
  "Other",
];

// Build categories list from base + any new ones in data
function buildCategories(transactions: { category: string }[]): string[] {
  const categorySet = new Set(BASE_CATEGORIES);
  for (const t of transactions) {
    categorySet.add(t.category);
  }
  return Array.from(categorySet);
}

// Build vocabulary from merchant names
function buildVocabulary(
  transactions: { merchant: string }[],
): Map<string, number> {
  const vocab = new Map<string, number>();
  vocab.set("<UNK>", 0);
  let idx = 1;
  for (const t of transactions) {
    const words = t.merchant.toLowerCase().split(/\s+/);
    for (const word of words) {
      if (!vocab.has(word)) {
        vocab.set(word, idx++);
      }
    }
  }
  return vocab;
}

// Convert merchant name to bag-of-words vector
function merchantToVector(
  merchant: string,
  vocab: Map<string, number>,
  vocabSize: number,
): number[] {
  const vector = new Array(vocabSize).fill(0);
  const words = merchant.toLowerCase().split(/\s+/);
  for (const word of words) {
    const idx = vocab.get(word) ?? 0;
    vector[idx] = 1;
  }
  return vector;
}

// Normalize amount (log scale for wide range)
function normalizeAmount(amount: number): number {
  return Math.log10(amount + 1) / 4; // log10(10000) ~ 4
}

async function train() {
  // Load training data
  const syntheticData = JSON.parse(
    fs.readFileSync("./data/synthetic-transactions.json", "utf-8"),
  );
  const userCorrections = JSON.parse(
    fs.readFileSync("./data/user-corrections.json", "utf-8"),
  );

  // Merge data, user corrections have higher weight (duplicated 50x)
  const allData = [...syntheticData];
  for (const correction of userCorrections) {
    // Weight user feedback heavily (50x) to override training patterns
    for (let i = 0; i < 50; i++) {
      allData.push(correction);
    }
  }

  console.log(
    `Training on ${syntheticData.length} synthetic + ${userCorrections.length} user corrections`,
  );

  // Build categories (base + any new ones from user data)
  const categories = buildCategories(allData);
  const newCategories = categories.filter((c) => !BASE_CATEGORIES.includes(c));
  if (newCategories.length > 0) {
    console.log(`New user-defined categories: ${newCategories.join(", ")}`);
  }
  console.log(`Total categories: ${categories.length}`);

  // Build vocabulary
  const vocab = buildVocabulary(allData);
  const vocabSize = vocab.size;
  console.log(`Vocabulary size: ${vocabSize}`);

  // Save vocabulary for inference
  fs.mkdirSync("./budget-model", { recursive: true });
  fs.writeFileSync(
    "./budget-model/vocab.json",
    JSON.stringify(Object.fromEntries(vocab)),
  );
  fs.writeFileSync(
    "./budget-model/categories.json",
    JSON.stringify(categories),
  );

  // Prepare features and labels
  const features: number[][] = [];
  const labels: number[] = [];

  for (const t of allData) {
    const merchantVec = merchantToVector(t.merchant, vocab, vocabSize);
    const amountNorm = normalizeAmount(t.amount);
    features.push([...merchantVec, amountNorm]);
    labels.push(categories.indexOf(t.category));
  }

  const xs = tf.tensor2d(features);
  const ys = tf.oneHot(tf.tensor1d(labels, "int32"), categories.length);

  // Build model
  const inputSize = vocabSize + 1; // vocab + amount
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 64, activation: "relu", inputShape: [inputSize] }),
  );
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(
    tf.layers.dense({ units: categories.length, activation: "softmax" }),
  );

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  console.log("\nTraining model...");
  await model.fit(xs, ys, {
    epochs: 100,
    batchSize: 16,
    validationSplit: 0.2,
    verbose: 0,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 20 === 0) {
          console.log(
            `  Epoch ${epoch + 1}: loss=${logs?.loss?.toFixed(4)}, accuracy=${logs?.acc?.toFixed(4)}`,
          );
        }
      },
    },
  });

  // Test predictions
  console.log("\nTest predictions:");
  const testCases = [
    { merchant: "Starbucks", amount: 5.5 },
    { merchant: "Amazon", amount: 299.0 },
    { merchant: "Shell Gas", amount: 45.0 },
    { merchant: "Netflix", amount: 15.99 },
    { merchant: "Kroger", amount: 85.0 },
    { merchant: "TJ Maxx", amount: 55.0 },
    { merchant: "City Utilities", amount: 200.0 },
    { merchant: "Farmers Dog", amount: 75.0 },
    { merchant: "Unknown Store", amount: 50.0 },
  ];

  for (const test of testCases) {
    const vec = [
      ...merchantToVector(test.merchant, vocab, vocabSize),
      normalizeAmount(test.amount),
    ];
    const pred = model.predict(tf.tensor2d([vec])) as tf.Tensor;
    const probs = await pred.data();
    const maxIdx = probs.indexOf(Math.max(...probs));
    const confidence = probs[maxIdx];
    console.log(
      `  ${test.merchant} $${test.amount} -> ${categories[maxIdx]} (${(confidence * 100).toFixed(1)}%)`,
    );
  }

  // Save model
  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      fs.writeFileSync(
        "./budget-model/model.json",
        JSON.stringify({
          modelTopology: artifacts.modelTopology,
          weightsManifest: [
            { paths: ["weights.bin"], weights: artifacts.weightSpecs },
          ],
        }),
      );
      if (artifacts.weightData) {
        fs.writeFileSync(
          "./budget-model/weights.bin",
          new Uint8Array(artifacts.weightData as ArrayBuffer),
        );
      }
      console.log("\nModel saved to ./budget-model/");
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
        },
      };
    }),
  );
}

train().catch(console.error);
