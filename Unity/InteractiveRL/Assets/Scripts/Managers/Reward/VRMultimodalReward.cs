// VRMultimodalReward.cs
// Experiment 3: VR human gives rewards via buttons, voice, and head gestures.

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Windows.Speech;
using UnityEngine.InputSystem;
using Utilities;

namespace Managers.Reward
{
    public class VRMultimodalReward : MonoBehaviour, IRewardProvider
    {
        [Header("Reward Values")]
        public float buttonReward = 1.00f;
        public float buttonPenalty = -1.00f;
        public float voiceReward = 0.75f;
        public float voicePenalty = -0.75f;
        public float nodReward = 0.50f;
        public float shakePenalty = -0.50f;

        [Header("Timing")]
        public float cooldownSeconds = 0.8f;

        [Header("VR Input Actions")]
        [Tooltip("Right controller grip = positive reward")]
        public InputActionReference rewardPlusAction;
        [Tooltip("Left controller grip = negative reward")]
        public InputActionReference rewardMinusAction;

        [Header("Keyboard Fallback")]
        public KeyCode rewardPlusKey = KeyCode.UpArrow;
        public KeyCode rewardMinusKey = KeyCode.DownArrow;

        [Header("References")]
        [Tooltip("XR Camera transform. Leave empty to auto-find.")]
        public Transform headTransform;

        [Header("Gesture Sensitivity")]
        [Tooltip("Min angular speed (deg/s) to register as intentional movement.")]
        public float gestureVelocityThreshold = 40f;
        [Tooltip("Peak angular speed (deg/s) that must be reached for gesture to count.")]
        public float minimumPeakVelocity = 60f;
        [Tooltip("Direction reversals needed to confirm a nod or shake. (1 nod cycle = 2 reversals)")]
        public int requiredReversals = 2;
        [Tooltip("Max time window (s) in which all reversals must occur.")]
        public float gestureWindowSeconds = 1.4f;

#if UNITY_EDITOR
        [Header("Debug (Editor Only)")]
        public bool debugOverlay = false;
#endif

        // IRewardProvider
        public bool IsEnabled { get; set; } = true;
        public UnityEvent<float> OnReward { get; } = new UnityEvent<float>();

        // Voice
        private readonly string[] _goodWords = { "good", "yes", "great", "nice", "perfect" };
        private readonly string[] _badWords = { "bad", "no", "wrong", "incorrect", "stop" };
        private KeywordRecognizer _keywordRecognizer;

        // Shared state 
        private float _lastRewardTime = -999f;

        // Head gesture
        private Vector3 _lastHeadEuler;
        private GestureAxisState _pitchState; // nod   (X axis)
        private GestureAxisState _yawState;   // shake (Y axis)

        private struct GestureAxisState
        {
            public bool Active;
            public float LastSign;
            public float PeakVelocity;
            public int Reversals;
            public float WindowStartTime;
            public float LastReversalTime;

            public void Clear() => this = default;
        }

        // Unity lifecycle
        private void OnEnable()
        {
            rewardPlusAction?.action.Enable();
            rewardMinusAction?.action.Enable();
        }

        private void OnDisable()
        {
            rewardPlusAction?.action.Disable();
            rewardMinusAction?.action.Disable();
        }

        private void Start()
        {
            SetupHeadTracking();
            SetupVoiceRecognition();
        }

        private void Update()
        {
            if (!IsEnabled) return;
            CheckButtons();
            CheckHeadGesture();
        }

        private void OnDestroy() => TearDownVoiceRecognition();

        // Setup
        private void SetupHeadTracking()
        {
            if (headTransform == null)
            {
                var xrOrigin = FindFirstObjectByType<Unity.XR.CoreUtils.XROrigin>();
                if (xrOrigin != null) headTransform = xrOrigin.Camera.transform;
            }

            if (headTransform != null)
                _lastHeadEuler = headTransform.eulerAngles;
            else
                Debug.LogWarning("[VRReward] No head transform found – gesture detection disabled.");
        }

        private void SetupVoiceRecognition()
        {
            var allWords = new List<string>(_goodWords);
            allWords.AddRange(_badWords);

            try
            {
                _keywordRecognizer = new KeywordRecognizer(allWords.ToArray());
                _keywordRecognizer.OnPhraseRecognized += OnVoiceCommand;
                _keywordRecognizer.Start();
                Debug.Log($"[VRReward] Voice recognition started ({allWords.Count} keywords).");
            }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"[VRReward] Voice recognition failed: {ex.Message}");
            }
        }

        private void TearDownVoiceRecognition()
        {
            if (_keywordRecognizer == null) return;
            if (_keywordRecognizer.IsRunning) _keywordRecognizer.Stop();
            _keywordRecognizer.OnPhraseRecognized -= OnVoiceCommand;
            _keywordRecognizer.Dispose();
            _keywordRecognizer = null;
        }

        // Buttons
        private void CheckButtons()
        {
            bool good = rewardPlusAction?.action.WasPressedThisFrame() ?? false;
            bool bad = rewardMinusAction?.action.WasPressedThisFrame() ?? false;

            if (Input.GetKeyDown(rewardPlusKey)) good = true;
            if (Input.GetKeyDown(rewardMinusKey)) bad = true;

            if (good) TryFireReward(buttonReward, "Button");
            else if (bad) TryFireReward(buttonPenalty, "Button");
        }

        // Voice

        private void OnVoiceCommand(PhraseRecognizedEventArgs args)
        {
            if (!IsEnabled) return;

            string text = args.text.ToLower();

            foreach (string word in _goodWords)
            {
                if (text.Contains(word))
                {
                    TryFireReward(voiceReward, $"Voice:'{text}'");
                    return;
                }
            }

            foreach (string word in _badWords)
            {
                if (text.Contains(word))
                {
                    TryFireReward(voicePenalty, $"Voice:'{text}'");
                    return;
                }
            }
        }

        // Head gesture
        private void CheckHeadGesture()
        {
            if (headTransform == null) return;

            float dt = Time.deltaTime;
            if (dt <= Mathf.Epsilon) { _lastHeadEuler = headTransform.eulerAngles; return; }

            Vector3 current = headTransform.eulerAngles;

            float pitchVel = Mathf.DeltaAngle(_lastHeadEuler.x, current.x) / dt; // nod
            float yawVel = Mathf.DeltaAngle(_lastHeadEuler.y, current.y) / dt; // shake

            ProcessAxis(pitchVel, ref _pitchState, isNod: true);
            ProcessAxis(yawVel, ref _yawState, isNod: false);

            _lastHeadEuler = current;
        }

        // Counts direction reversals on one axis. Fires when enough reversals
        // happen within the time window and peak speed is sufficient.
        private void ProcessAxis(float velocity, ref GestureAxisState state, bool isNod)
        {
            bool moving = Mathf.Abs(velocity) >= gestureVelocityThreshold;

            if (!moving)
            {
                if (state.Active && Time.time - state.LastReversalTime > gestureWindowSeconds * 0.5f)
                    state.Clear();
                return;
            }

            float sign = Mathf.Sign(velocity);

            if (!state.Active)
            {
                state.Active = true;
                state.WindowStartTime = Time.time;
                state.LastReversalTime = Time.time;
                state.LastSign = sign;
                state.PeakVelocity = Mathf.Abs(velocity);
                return;
            }

            state.PeakVelocity = Mathf.Max(state.PeakVelocity, Mathf.Abs(velocity));

            if (Time.time - state.WindowStartTime > gestureWindowSeconds) { state.Clear(); return; }

            if (sign != state.LastSign)
            {
                state.Reversals++;
                state.LastSign = sign;
                state.LastReversalTime = Time.time;
            }

            if (state.Reversals >= requiredReversals)
            {
                if (state.PeakVelocity >= minimumPeakVelocity)
                    TryFireReward(isNod ? nodReward : shakePenalty, isNod ? "Gesture:Nod" : "Gesture:Shake");

                state.Clear();
            }
        }

        // Shared reward dispatch

        private void TryFireReward(float value, string source)
        {
            if (!IsEnabled) return;
            if (Time.time - _lastRewardTime < cooldownSeconds) return;

            _lastRewardTime = Time.time;
            FeedbackLogger.Add(source, value);
            OnReward?.Invoke(value);
            Debug.Log($"[VRReward] {source} → {value:+0.##;-0.##;0}");
        }

        // Public API

        public void Reset()
        {
            _lastRewardTime = -999f;
            _pitchState.Clear();
            _yawState.Clear();

            if (_keywordRecognizer != null && _keywordRecognizer.IsRunning)
            {
                _keywordRecognizer.Stop();
                _keywordRecognizer.Start();
            }
        }

#if UNITY_EDITOR
        private void OnGUI()
        {
            if (!debugOverlay || headTransform == null) return;

            float dt = Time.deltaTime;
            if (dt <= Mathf.Epsilon) return;

            Vector3 cur = headTransform.eulerAngles;
            float pv = Mathf.DeltaAngle(_lastHeadEuler.x, cur.x) / dt;
            float yv = Mathf.DeltaAngle(_lastHeadEuler.y, cur.y) / dt;

            GUILayout.BeginArea(new Rect(10, 10, 340, 100));
            GUILayout.Label("<b>[VRReward Debug]</b>");
            GUILayout.Label($"Pitch (nod):  {pv:+000.0;-000.0} deg/s | rev={_pitchState.Reversals} peak={_pitchState.PeakVelocity:000.0}");
            GUILayout.Label($"Yaw (shake): {yv:+000.0;-000.0} deg/s | rev={_yawState.Reversals} peak={_yawState.PeakVelocity:000.0}");
            GUILayout.Label($"Cooldown: {Mathf.Max(0, cooldownSeconds - (Time.time - _lastRewardTime)):0.00}s");
            GUILayout.EndArea();
        }
#endif
    }
}
