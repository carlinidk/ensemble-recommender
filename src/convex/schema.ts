import { authTables } from "@convex-dev/auth/server";
import { defineSchema, defineTable } from "convex/server";
import { Infer, v } from "convex/values";

// default user roles. can add / remove based on the project as needed
export const ROLES = {
  ADMIN: "admin",
  USER: "user",
  MEMBER: "member",
} as const;

export const roleValidator = v.union(
  v.literal(ROLES.ADMIN),
  v.literal(ROLES.USER),
  v.literal(ROLES.MEMBER),
);
export type Role = Infer<typeof roleValidator>;

const schema = defineSchema(
  {
    // default auth tables using convex auth.
    ...authTables, // do not remove or modify

    // the users table is the default users table that is brought in by the authTables
    users: defineTable({
      name: v.optional(v.string()), // name of the user. do not remove
      image: v.optional(v.string()), // image of the user. do not remove
      email: v.optional(v.string()), // email of the user. do not remove
      emailVerificationTime: v.optional(v.number()), // email verification time. do not remove
      isAnonymous: v.optional(v.boolean()), // is the user anonymous. do not remove

      role: v.optional(roleValidator), // role of the user. do not remove
    }).index("email", ["email"]), // index for the email. do not remove or modify

    // A signed-in user's own ratings. These form their taste profile and are
    // excluded from every recommendation candidate list.
    ratings: defineTable({
      userId: v.id("users"),
      movieId: v.number(),
      rating: v.number(),
      ratedAt: v.number(),
    })
      .index("by_user", ["userId"])
      .index("by_user_movie", ["userId", "movieId"]),

    // Serving preferences: Top-K size, ranking mode and enhancement weights.
    profiles: defineTable({
      userId: v.id("users"),
      topK: v.number(),
      mode: v.string(),
      diversityLambda: v.number(),
      affinityWeight: v.number(),
      updatedAt: v.number(),
    }).index("by_user", ["userId"]),

    // Model registry metadata. Artifacts live in memory/filesystem, not in the
    // document store — only parameters, metrics and provenance are recorded.
    modelRuns: defineTable({
      userId: v.id("users"),
      modelId: v.string(),
      version: v.string(),
      algorithm: v.string(),
      trainingDate: v.number(),
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
    }).index("by_user", ["userId"]),

    // One row per stacking configuration per run, so configurations can be
    // compared without re-parsing the run document.
    experimentResults: defineTable({
      runId: v.id("modelRuns"),
      userId: v.id("users"),
      configId: v.string(),
      label: v.string(),
      baseLearners: v.array(v.string()),
      metaLearner: v.string(),
      mae: v.number(),
      mse: v.number(),
      rmse: v.number(),
      rank: v.number(),
      createdAt: v.number(),
    })
      .index("by_run", ["runId"])
      .index("by_user", ["userId"]),
  },
  {
    schemaValidation: false,
  },
);

export default schema;
