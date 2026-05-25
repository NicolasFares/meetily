-- Migration: Add support for the remote OpenAI-compatible transcription provider
-- Adds a base URL column for the remote endpoint and a dedicated API key column.

ALTER TABLE transcript_settings ADD COLUMN baseUrl TEXT;
ALTER TABLE transcript_settings ADD COLUMN remoteOpenAIApiKey TEXT;
