#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
# if defined(AZUL_BUILD_DLL)
#  define AZUL_API __declspec(dllexport)
# else
#  define AZUL_API __declspec(dllimport)
# endif
#else
# define AZUL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif
/* Three-player ABI. Kept separate so existing 2P checkpoints and clients remain binary compatible. */
enum { AZUL3_ACTION_COUNT = 240, AZUL3_OBSERVATION_SIZE = 244, AZUL3_ABI_VERSION = 1 };
typedef struct Azul3Batch Azul3Batch;
typedef struct Azul3StepResult { float reward; uint16_t scores[3]; uint8_t actor, terminated, round_finished, invalid_action; } Azul3StepResult;
AZUL_API int azul3_abi_version(void);
AZUL_API Azul3Batch* azul3_batch_create(size_t n, unsigned threads, uint64_t base_seed);
AZUL_API void azul3_batch_destroy(Azul3Batch* batch);
AZUL_API size_t azul3_batch_size(const Azul3Batch* batch);
AZUL_API int azul3_batch_reset(Azul3Batch* batch, uint64_t base_seed);
AZUL_API int azul3_batch_reset_at(Azul3Batch* batch, size_t index, uint64_t seed, unsigned starting_player);
AZUL_API int azul3_batch_observe(Azul3Batch* batch, float* observations, uint8_t* masks, uint8_t* players);
AZUL_API int azul3_batch_step(Azul3Batch* batch, const uint16_t* actions, Azul3StepResult* results);
AZUL_API int azul3_batch_import_public(Azul3Batch* batch, size_t index,
    const uint8_t* sources, const uint32_t* walls, const uint16_t* scores,
    const uint8_t* pattern_colors, const uint8_t* pattern_counts,
    const uint8_t* floor_tiles, const uint8_t* floor_counts,
    const uint8_t* bag, const uint8_t* discard, uint32_t round,
    uint8_t current, uint8_t next_start, uint8_t token_available);
typedef struct Azul3Search Azul3Search;
typedef struct Azul3SearchStats { uint64_t simulations, evaluations, chance_draws, chance_reuses, chance_duplicates, nodes; } Azul3SearchStats;
AZUL_API int azul3_search_abi_version(void);
AZUL_API Azul3Search* azul3_search_create(size_t n, unsigned threads, const struct AzulSearchConfig* config);
AZUL_API void azul3_search_destroy(Azul3Search* search);
AZUL_API int azul3_search_begin(Azul3Search* search, const Azul3Batch* batch, uint64_t seed);
AZUL_API int azul3_search_request(Azul3Search* search, float* observations, uint8_t* ready);
/* submit consumes logits[n][240] and values[n][3]. Values are in the requested
   observation's rotated player order: current player, next player, third player. */
AZUL_API int azul3_search_submit(Azul3Search* search, const float* logits, const float* values);
AZUL_API int azul3_search_results(Azul3Search* search, float* policies, uint16_t* actions, float* values, Azul3SearchStats* stats);
enum { AZUL_ACTION_COUNT = 180, AZUL_OBSERVATION_SIZE = 172, AZUL_ABI_VERSION = 1 };
typedef struct AzulBatch AzulBatch;
typedef struct AzulStepResult {
    float reward;             /* terminal +/-1 or 0, from actor's perspective */
    uint16_t scores[2];       /* absolute player order, after settlement */
    uint8_t actor;            /* player who was asked to take the action */
    uint8_t terminated;
    uint8_t round_finished;
    uint8_t invalid_action;   /* invalid action leaves the entire state unchanged */
} AzulStepResult;
AZUL_API int azul_abi_version(void);
/* threads=0: hardware concurrency; capped to batch size. n must be positive.
   Allocation and thread creation only here; no shared mutable game/RNG state.
   The same handle is NOT reentrant; different handles may be called concurrently. */
AZUL_API AzulBatch* azul_batch_create(size_t n, unsigned threads, uint64_t base_seed);
AZUL_API void azul_batch_destroy(AzulBatch* batch);
AZUL_API size_t azul_batch_size(const AzulBatch* batch);
/* C functions return 0 on success, -1 on invalid arguments/internal exception.
   Per-environment invalid actions are reported separately in results. */
AZUL_API int azul_batch_reset(AzulBatch* batch, uint64_t base_seed);
AZUL_API int azul_batch_reset_at(AzulBatch* batch, size_t index, uint64_t seed, unsigned starting_player);
/* Nullable output arrays: observations[n][172], masks[n][180], players[n].
   Observation perspective is each environment's current player. */
AZUL_API int azul_batch_observe(AzulBatch* batch, float* observations, uint8_t* masks, uint8_t* players);
/* actions[n], optional results[n]. Terminal states stay terminal until reset. */
AZUL_API int azul_batch_step(AzulBatch* batch, const uint16_t* actions, AzulStepResult* results);
/* Process-local snapshot, includes RNG, private to this exact library build.
   For long-term portable replay store seed, starting player and action IDs. */
AZUL_API size_t azul_snapshot_size(void);
AZUL_API int azul_batch_snapshot(AzulBatch* batch, size_t index, void* buffer, size_t bytes);
AZUL_API int azul_batch_restore(AzulBatch* batch, size_t index, const void* buffer, size_t bytes);
/* Full Gumbel search ABI 1. Calls on each handle must be sequential.
   begin copies the environment states; search NEVER advances the actual games.
   request writes one leaf per root into observations[n][172], ready[n] (0/1).
   submit consumes logits[n][180], values[n] only for ready roots. Repeat until
   request returns 0 ready roots. Root evaluation is outside simulation budget. */
typedef struct AzulSearch AzulSearch;
typedef struct AzulSearchConfig {
    int32_t simulations, candidates, max_depth, chance_initial, chance_cap;
    float gumbel_scale, value_scale, maxvisit_init;
} AzulSearchConfig;
typedef struct AzulSearchStats {
    uint64_t simulations, evaluations, chance_draws, chance_reuses, chance_duplicates, nodes;
} AzulSearchStats;
AZUL_API int azul_search_abi_version(void);
AZUL_API AzulSearch* azul_search_create(size_t n, unsigned threads, const AzulSearchConfig* config);
AZUL_API void azul_search_destroy(AzulSearch* search);
AZUL_API int azul_search_begin(AzulSearch* search, const AzulBatch* batch, uint64_t seed);
AZUL_API int azul_search_request(AzulSearch* search, float* observations, uint8_t* ready);
AZUL_API int azul_search_submit(AzulSearch* search, const float* logits, const float* values);
AZUL_API int azul_search_results(AzulSearch* search, float* policies, uint16_t* actions, float* values, AzulSearchStats* stats);
typedef struct AzulSearchTuning {
    int32_t q_mode, subtree_reuse, cache_capacity, symmetric_cache;
    float q_floor, chance_coefficient, chance_exponent, chance_sensitivity, afterstate_prior;
} AzulSearchTuning;
AZUL_API int azul_search_configure(AzulSearch* search, const AzulSearchTuning* tuning);
/* range roots use global index when deriving RNG streams; active may be null.
   model_version must change on ANY weight/precision/configuration publication. */
AZUL_API int azul_search_begin_range(AzulSearch* search, const AzulBatch* batch, size_t offset,
    uint64_t seed, uint64_t model_version, const uint8_t* active);
/* Returns counters: cache_hits, cache_misses, reused_nodes, afterstate_evaluations. */
AZUL_API int azul_search_diagnostics(AzulSearch* search, uint64_t* counters);
/* Explicit pre-deal observations for afterstate auxiliary learning. */
AZUL_API int azul_batch_step_afterstates(AzulBatch* batch, const uint16_t* actions, AzulStepResult* results,
    float* afterstates, uint8_t* flags, uint8_t* perspectives);
/* Manual deal at chance nodes. colors[20] is factory-major, 0..4 colors.
   The call consumes bag/discard according to normal rules and fills 5x4 tiles. */
AZUL_API int azul_batch_manual_deal(AzulBatch* batch, const uint8_t* colors);
AZUL_API int azul_manual_reset(AzulBatch* batch, uint64_t seed);
AZUL_API int azul_manual_step(AzulBatch* batch, uint16_t action, AzulStepResult* result);
/* Import one validated public draft position. Arrays are row-major:
   sources[6][5], walls[2] bitboards, patterns/floors[2][5], bag/discard[5]. */
AZUL_API int azul_batch_import_public(AzulBatch* batch, size_t index,
    const uint8_t* sources, const uint32_t* walls, const uint16_t* scores,
    const uint8_t* pattern_colors, const uint8_t* pattern_counts,
    const uint8_t* floor_tiles, const uint8_t* floor_counts,
    const uint8_t* bag, const uint8_t* discard, uint32_t round,
    uint8_t current, uint8_t next_start, uint8_t token_available);
#ifdef __cplusplus
}
#endif
