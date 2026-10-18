
import json

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="DrishtiPay Lite",
    page_icon="🔊",
    layout="centered",
)

st.title("🔊 DrishtiPay Lite")
st.caption("Voice-guided currency accessibility prototype")

st.warning(
    "Prototype only. This version does not identify real "
    "banknote denominations or verify genuine currency."
)


def speech_button(message, button_label="🔊 Speak instruction"):
    """Show a browser speech button with a user-triggered action."""
    message_js = json.dumps(message)

    components.html(
        f"""
        <div style="font-family:Arial,sans-serif;padding:6px 0;">
          <button id="speakBtn"
            style="font-size:17px;padding:12px 16px;
                   border-radius:10px;cursor:pointer;">
            {button_label}
          </button>
          <p id="speechStatus" aria-live="polite"
             style="font-size:14px;"></p>
        </div>
        <script>
          const textToSpeak = {message_js};
          const btn = document.getElementById("speakBtn");
          const status = document.getElementById("speechStatus");

          btn.addEventListener("click", function () {{
            if (!("speechSynthesis" in window)) {{
              status.textContent =
                "Speech is not supported in this browser.";
              return;
            }}

            window.speechSynthesis.cancel();
            const utterance =
              new SpeechSynthesisUtterance(textToSpeak);
            utterance.lang = "en-IN";

            utterance.onstart = function () {{
              status.textContent = "Speaking...";
            }};
            utterance.onend = function () {{
              status.textContent = "Speech finished.";
            }};
            utterance.onerror = function () {{
              status.textContent =
                "Speech failed. Check phone volume and browser settings.";
            }};

            window.speechSynthesis.speak(utterance);
            status.textContent = "Speech requested.";
          }});
        </script>
        """,
        height=100,
    )


st.header("1. Test voice")
st.write("Tap this button first to check whether your phone can speak.")
speech_button(
    "Welcome to DrishtiPay. Voice feedback test successful.",
    "🔊 Test voice",
)

st.divider()
st.header("2. Choose a mode")

mode = st.radio(
    "Select a test",
    ["Guided Camera Test", "Demo Mode — SIMULATED"],
)

if mode == "Guided Camera Test":
    st.info(
        "Place one banknote on a flat surface with good lighting. "
        "This test captures a still photo; it does not recognise "
        "the denomination."
    )

    st.caption(
        "If the camera does not open on your phone, see the "
        "camera troubleshooting instructions below."
    )

    photo = st.camera_input("Capture a banknote")

    if photo is None:
        camera_message = (
            "No photo has been captured yet. Allow camera access "
            "and try taking a photo."
        )
        st.write(camera_message)
        speech_button(camera_message)
    else:
        st.image(
            photo,
            caption="Captured photo — no denomination prediction",
            use_container_width=True,
        )

        camera_message = (
            "Photo captured successfully. DrishtiPay has not "
            "identified the banknote denomination."
        )
        st.success(camera_message)
        speech_button(camera_message)

elif mode == "Demo Mode — SIMULATED":
    st.subheader("🧪 Demo Mode")
    st.write(
        "This is a simulation. The amount is selected by you, "
        "not detected by the camera or AI."
    )

    scenario = st.selectbox(
        "Choose a demo scenario",
        [
            "Welcome",
            "Reposition the phone",
            "Note is centred (simulation)",
            "Unclear result",
            "Simulated denomination",
        ],
    )

    messages = {
        "Welcome":
            "Welcome to DrishtiPay. This is a simulated demo.",
        "Reposition the phone":
            "Demo instruction. Move the phone slightly to the right.",
        "Note is centred (simulation)":
            "Demo instruction. The note is centred. This is a simulation.",
        "Unclear result":
            "Demo result. I cannot identify this note reliably. Please try again.",
    }

    if scenario == "Simulated denomination":
        amount = st.selectbox(
            "Choose a simulated amount",
            [10, 20, 50, 100, 200, 500],
        )
        message = (
            f"Demo only. Simulated result: {amount} rupees. "
            "This amount was not detected from an image."
        )
    else:
        message = messages[scenario]

    st.info("SIMULATED OUTPUT: " + message)
    speech_button(message, "🔊 Speak demo result")

st.divider()
st.caption(
    "Do not rely on this prototype for payments. "
    "It cannot authenticate currency."
)
