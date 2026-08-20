using System.Collections.Generic;
using UnityEngine.Events;

namespace Utilities
{
    /// Static feedback log. Any reward provider calls FeedbackLogger.Add(source, value).
    /// DataDisplay reads GetEntries() every frame to update the UI.
    public static class FeedbackLogger
    {
        public const int MaxEntries = 5;

        public struct Entry
        {
            public string source; // e.g. "Button", "Voice", "Gesture", "Keyboard", "Auto"
            public float  value;
        }

        private static readonly Queue<Entry> logEntries = new Queue<Entry>(MaxEntries + 1);

        // Fired whenever a new entry is added
        public static UnityEvent<Entry> OnNewEntry { get; } = new UnityEvent<Entry>();

        public static void Add(string source, float value)
        {
            var entry = new Entry { source = source, value = value };

            logEntries.Enqueue(entry);
            while (logEntries.Count > MaxEntries)
                logEntries.Dequeue();

            OnNewEntry.Invoke(entry);
        }

        /// Returns entries oldest-first so the UI can print them top-to-bottom.
        public static Entry[] GetEntries() => logEntries.ToArray();

        public static void Clear() => logEntries.Clear();
    }
}