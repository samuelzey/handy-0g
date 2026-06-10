import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface CloudAsrFallbackToLocalProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * If a cloud ASR call fails (network error, 5xx, attestation rejected),
 * silently retry on the local engine instead of returning an error.
 *
 * On by default so a transient 0G outage never eats a user's recording.
 * Users who explicitly do not want a fallback (e.g. for compliance —
 * audio should never decrypt outside a TEE) can flip this off and pair
 * it with `require_tee_proof = true`.
 */
export const CloudAsrFallbackToLocal: React.FC<CloudAsrFallbackToLocalProps> =
  React.memo(({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    // Default-true semantics: missing setting means fallback is enabled.
    const fallback = getSetting("cloud_asr_fallback_to_local") ?? true;
    const cloudEnabled = getSetting("cloud_asr_enabled") || false;

    return (
      <ToggleSwitch
        checked={fallback}
        onChange={(v) => updateSetting("cloud_asr_fallback_to_local", v)}
        isUpdating={isUpdating("cloud_asr_fallback_to_local")}
        label={t("settings.cloudAsr.fallback.label")}
        description={t("settings.cloudAsr.fallback.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
        disabled={!cloudEnabled}
      />
    );
  });
