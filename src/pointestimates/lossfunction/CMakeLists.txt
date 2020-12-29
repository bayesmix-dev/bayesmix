target_sources(bayesmix
        PUBLIC
        BinderLoss.cpp
        BinderLoss.hpp
        LossFunction.cpp
        LossFunction.hpp
        VariationInformation.cpp
        VariationInformation.hpp
)

add_executable(loss $<TARGET_OBJECTS:bayesmix> loss.cpp)
