import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface CloudAsrToggleProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * Master toggle for routing audio through 0G Compute Router
 * (Whisper-large-v3 inside an Intel TDX + H100/H200 TEE) instead of the
 * local whisper.cpp / Parakeet engine.
 *
 * When this is on, every push-to-talk recording is uploaded; users must
 * configure the `zerog` provider API key in Post Processing settings for
 * the call to succeed. See actions.rs for the runtime fallback logic.
 */
export const CloudAsrToggle: React.FC<CloudAsrToggleProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const enabled = getSetting("cloud_asr_enabled") || false;

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(v) => updateSetting("cloud_asr_enabled", v)}
        isUpdating={isUpdating("cloud_asr_enabled")}
        label={t("settings.cloudAsr.enabled.label")}
        description={t("settings.cloudAsr.enabled.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  },
);
