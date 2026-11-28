#include "common/orderedcompletions.h"

#include <gtest/gtest.h>

#include <string>
#include <variant>

namespace
{
    using Completions = rsinfer::OrderedCompletions<int>;

    int take_value(Completions& completions)
    {
        auto completion = completions.take();
        EXPECT_TRUE(completion.has_value());
        EXPECT_TRUE(std::holds_alternative<int>(*completion));
        return std::get<int>(*completion);
    }
}

TEST(OrderedCompletions, EmptyIsReadyAndHasNothing)
{
    Completions completions;
    EXPECT_TRUE(completions.ready());
    EXPECT_FALSE(completions.take().has_value());
}

TEST(OrderedCompletions, OutOfOrderResultsComeBackInSubmissionOrder)
{
    Completions completions;
    const auto first = completions.reserve();
    const auto second = completions.reserve();
    const auto third = completions.reserve();

    completions.complete(third, 3);
    completions.complete(second, 2);
    // The oldest request is still running: nothing may be handed out yet.
    EXPECT_FALSE(completions.ready());

    completions.complete(first, 1);
    ASSERT_TRUE(completions.ready());
    EXPECT_EQ(take_value(completions), 1);
    EXPECT_EQ(take_value(completions), 2);
    EXPECT_EQ(take_value(completions), 3);
    EXPECT_FALSE(completions.take().has_value());
}

TEST(OrderedCompletions, CancelledTicketsAreSkipped)
{
    Completions completions;
    const auto first = completions.reserve();
    const auto second = completions.reserve();
    const auto third = completions.reserve();

    completions.complete(third, 3);
    completions.cancel(first);
    EXPECT_FALSE(completions.ready()); // second is still pending
    completions.cancel(second);

    ASSERT_TRUE(completions.ready());
    EXPECT_EQ(take_value(completions), 3);
    EXPECT_TRUE(completions.ready());
    EXPECT_FALSE(completions.take().has_value());
}

TEST(OrderedCompletions, ErrorsKeepTheirPlace)
{
    Completions completions;
    const auto first = completions.reserve();
    const auto second = completions.reserve();
    completions.complete(second, 2);
    completions.complete(first, std::string("failed"));

    auto error = completions.take();
    ASSERT_TRUE(error.has_value());
    ASSERT_TRUE(std::holds_alternative<std::string>(*error));
    EXPECT_EQ(std::get<std::string>(*error), "failed");
    EXPECT_EQ(take_value(completions), 2);
}

TEST(OrderedCompletions, AllCancelledMeansNothingPending)
{
    Completions completions;
    completions.cancel(completions.reserve());
    EXPECT_TRUE(completions.ready());
    EXPECT_FALSE(completions.take().has_value());

    // Tickets keep counting after cancelled ones.
    const auto next = completions.reserve();
    EXPECT_FALSE(completions.ready());
    completions.complete(next, 7);
    EXPECT_EQ(take_value(completions), 7);
}
