import { getAuthUserId } from "@convex-dev/auth/server";
import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

const RATING_MIN = 1;
const RATING_MAX = 5;

/**
 * Persistence layer for the recommender.
 *
 * Training happens in an offline browser worker; what lands in the database is
 * the user's ratings, their serving preferences, and the metadata + metrics of
 * every training run. Trained artifacts are intentionally NOT stored in
 * document blobs (they are large, binary-ish and re-derivable from the recorded
 * configuration), which mirrors the "artifacts on disk, metadata in the DB"
 * split from the project brief.
 */

export const myRatings = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return [];
    const rows = await ctx.db
      .query("ratings")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .collect();
    return rows.map((row) => ({
      movieId: row.movieId,
      rating: row.rating,
      ratedAt: row.ratedAt,
    }));
  },
});

export const myProfile = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return null;
    return await ctx.db
      .query("profiles")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .first();
  },
});

export const latestRun = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return null;
    return await ctx.db
      .query("modelRuns")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .order("desc")
      .first();
  },
});

export const runExperiments = query({
  args: { runId: v.id("modelRuns") },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return [];
    return await ctx.db
      .query("experimentResults")
      .withIndex("by_run", (q) => q.eq("runId", args.runId))
      .collect();
  },
});

export const rateMovie = mutation({
  args: { movieId: v.number(), rating: v.number() },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Not signed in.");
    if (!Number.isInteger(args.movieId) || args.movieId <= 0) {
      throw new Error("movieId must be a positive integer.");
    }
    if (args.rating < RATING_MIN || args.rating > RATING_MAX) {
      throw new Error(`rating must be between ${RATING_MIN} and ${RATING_MAX}.`);
    }
    const existing = await ctx.db
      .query("ratings")
      .withIndex("by_user_movie", (q) => q.eq("userId", userId).eq("movieId", args.movieId))
      .first();
    const ratedAt = Date.now();
    if (existing) {
      await ctx.db.patch(existing._id, { rating: args.rating, ratedAt });
      return existing._id;
    }
    return await ctx.db.insert("ratings", {
      userId,
      movieId: args.movieId,
      rating: args.rating,
      ratedAt,
    });
  },
});

export const clearMovieRating = mutation({
  args: { movieId: v.number() },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Not signed in.");
    const existing = await ctx.db
      .query("ratings")
      .withIndex("by_user_movie", (q) => q.eq("userId", userId).eq("movieId", args.movieId))
      .first();
    if (existing) await ctx.db.delete(existing._id);
  },
});

export const clearRatings = mutation({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Not signed in.");
    const rows = await ctx.db
      .query("ratings")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .collect();
    for (const row of rows) await ctx.db.delete(row._id);
    return rows.length;
  },
});

export const saveProfile = mutation({
  args: {
    topK: v.number(),
    mode: v.string(),
    diversityLambda: v.number(),
    affinityWeight: v.number(),
  },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Not signed in.");
    if (args.mode !== "accuracy" && args.mode !== "diversity") {
      throw new Error("mode must be 'accuracy' or 'diversity'.");
    }
    const topK = Math.max(1, Math.min(100, Math.round(args.topK)));
    const diversityLambda = Math.max(0, Math.min(1, args.diversityLambda));
    const affinityWeight = Math.max(0, Math.min(1, args.affinityWeight));
    const payload = {
      topK,
      mode: args.mode,
      diversityLambda,
      affinityWeight,
      updatedAt: Date.now(),
    };
    const existing = await ctx.db
      .query("profiles")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .first();
    if (existing) {
      await ctx.db.patch(existing._id, payload);
      return existing._id;
    }
    return await ctx.db.insert("profiles", { userId, ...payload });
  },
});

export const recordRun = mutation({
  args: {
    modelId: v.string(),
    version: v.string(),
    algorithm: v.string(),
    datasetVersion: v.string(),
    featureVersion: v.string(),
    randomSeed: v.number(),
    cvFolds: v.number(),
    datasetScope: v.string(),
    splitStrategy: v.string(),
    searchMode: v.string(),
    hyperparameters: v.any(),
    metrics: v.any(),
    standalone: v.any(),
    stacking: v.any(),
    trainingSeconds: v.number(),
    nTrain: v.number(),
    nTest: v.number(),
  },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Not signed in.");
    const runId = await ctx.db.insert("modelRuns", {
      userId,
      modelId: args.modelId,
      version: args.version,
      algorithm: args.algorithm,
      trainingDate: Date.now(),
      datasetVersion: args.datasetVersion,
      featureVersion: args.featureVersion,
      randomSeed: args.randomSeed,
      cvFolds: args.cvFolds,
      datasetScope: args.datasetScope,
      splitStrategy: args.splitStrategy,
      searchMode: args.searchMode,
      hyperparameters: args.hyperparameters,
      metrics: args.metrics,
      standalone: args.standalone,
      stacking: args.stacking,
      trainingSeconds: args.trainingSeconds,
      nTrain: args.nTrain,
      nTest: args.nTest,
    });

    const stacking = Array.isArray(args.stacking) ? args.stacking : [];
    const createdAt = Date.now();
    for (const row of stacking as {
      id?: string;
      label?: string;
      baseNames?: string[];
      metaName?: string;
      mae?: number;
      mse?: number;
      rmse?: number;
      rank?: number;
    }[]) {
      await ctx.db.insert("experimentResults", {
        runId,
        userId,
        configId: String(row.id ?? "?"),
        label: String(row.label ?? "unknown"),
        baseLearners: Array.isArray(row.baseNames) ? row.baseNames.map(String) : [],
        metaLearner: String(row.metaName ?? "unknown"),
        mae: Number(row.mae ?? 0),
        mse: Number(row.mse ?? 0),
        rmse: Number(row.rmse ?? 0),
        rank: Number(row.rank ?? 0),
        createdAt,
      });
    }

    return runId;
  },
});
