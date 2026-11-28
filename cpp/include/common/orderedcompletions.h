#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace rsinfer
{
    //! Results of inference requests, handed out in submission order.
    //!
    //! Each request takes a ticket when it is submitted (reserve) and later
    //! either completes it with a result or an error message (complete), or
    //! cancels it when the error was reported to the submitter directly
    //! (cancel). take() returns completions strictly in ticket order, skipping
    //! cancelled tickets, so get_results matches requests to results even when
    //! several execution contexts finish out of order.
    //!
    //! Not thread-safe: the owner guards every call with its own mutex.
    template <class Result>
    class OrderedCompletions
    {
    public:
        using Completion = std::variant<Result, std::string>;

        uint64_t reserve()
        {
            ++_pending;
            return _next_ticket++;
        }

        void complete(uint64_t ticket, Completion completion)
        {
            _done.insert_or_assign(ticket, std::optional<Completion>(std::move(completion)));
            --_pending;
        }

        void cancel(uint64_t ticket)
        {
            _done.insert_or_assign(ticket, std::nullopt);
            --_pending;
            skip_cancelled();
        }

        //! True when take() can return the next result, or when nothing is
        //! pending and take() will report that there is no result.
        bool ready() const
        {
            return _done.contains(_next_deliver) || _pending == 0;
        }

        //! The next result in submission order, or nullopt when none is pending.
        std::optional<Completion> take()
        {
            auto it = _done.find(_next_deliver);
            if (it == _done.end())
                return std::nullopt;
            Completion completion = std::move(*it->second);
            _done.erase(it);
            ++_next_deliver;
            skip_cancelled();
            return completion;
        }

    private:
        void skip_cancelled()
        {
            for (auto it = _done.find(_next_deliver); it != _done.end() && !it->second; it = _done.find(_next_deliver))
            {
                _done.erase(it);
                ++_next_deliver;
            }
        }

        std::map<uint64_t, std::optional<Completion>> _done;
        uint64_t _next_ticket{ 0 };
        uint64_t _next_deliver{ 0 };
        size_t _pending{ 0 };
    };
}
