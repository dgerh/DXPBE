#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=26, b0),"     /* 20 constants: w/ one int3, see PBMPM Constants */ \
"DescriptorTable(SRV(t0, numDescriptors=2)),"    /* SRVs for g_particles, g_particleCount */ \
"DescriptorTable(UAV(u0, numDescriptors=2)),"   /* UAVs for g_particleInsertCounters and g_particleData */ \
"DescriptorTable(SRV(t2, numDescriptors=1))" /* SRV for g_bukkitIndexStart */