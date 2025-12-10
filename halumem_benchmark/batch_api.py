"""OpenAI Batch API support for HaluMem benchmark.

Submits large workloads as batch jobs for:
- No rate limiting issues
- 50% cost reduction
- Reliable processing
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a submitted batch job."""
    batch_id: str
    input_file_id: str
    status: str
    request_count: int
    created_at: float


class BatchProcessor:
    """Process large workloads using OpenAI's Batch API.

    The Batch API allows submitting thousands of requests at once,
    with 50% cost savings and no rate limiting.

    Example:
        >>> processor = BatchProcessor(client)
        >>> batch_id = await processor.submit_extraction_batch(dialogues)
        >>> results = await processor.wait_for_completion(batch_id)
    """

    def __init__(self, client, output_dir: Path = None):
        """Initialize the batch processor.

        Args:
            client: OpenAI AsyncClient instance.
            output_dir: Directory for batch files. Defaults to temp dir.
        """
        self.client = client
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "halumem_batches"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_batch_request(
        self,
        custom_id: str,
        model: str,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Create a single batch request in JSONL format.

        Note: Temperature is not included as gpt-5-nano only supports default (1).
        """
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages
            }
        }

    async def submit_batch(
        self,
        requests: List[Dict[str, Any]],
        description: str = "HaluMem benchmark batch"
    ) -> BatchJob:
        """Submit a batch of requests.

        Args:
            requests: List of batch request dicts.
            description: Description for the batch job.

        Returns:
            BatchJob with batch_id and status.
        """
        # Write requests to JSONL file
        batch_file = self.output_dir / f"batch_input_{int(time.time())}.jsonl"
        with open(batch_file, 'w') as f:
            for req in requests:
                json.dump(req, f)
                f.write('\n')

        logger.info(f"Created batch file with {len(requests)} requests: {batch_file}")

        # Upload file
        with open(batch_file, 'rb') as f:
            file_response = await self.client.files.create(
                file=f,
                purpose="batch"
            )

        logger.info(f"Uploaded batch file: {file_response.id}")

        # Create batch job
        batch_response = await self.client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description}
        )

        logger.info(f"Created batch job: {batch_response.id}")

        return BatchJob(
            batch_id=batch_response.id,
            input_file_id=file_response.id,
            status=batch_response.status,
            request_count=len(requests),
            created_at=time.time()
        )

    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get the status of a batch job."""
        batch = await self.client.batches.retrieve(batch_id)
        return {
            "status": batch.status,
            "request_counts": batch.request_counts,
            "created_at": batch.created_at,
            "completed_at": getattr(batch, 'completed_at', None),
            "output_file_id": getattr(batch, 'output_file_id', None),
            "error_file_id": getattr(batch, 'error_file_id', None)
        }

    async def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: int = 30,
        timeout: int = 86400  # 24 hours
    ) -> Dict[str, str]:
        """Wait for a batch to complete and return results.

        Args:
            batch_id: The batch job ID.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait.

        Returns:
            Dict mapping custom_id to response content.
        """
        start_time = time.time()

        while True:
            status = await self.get_batch_status(batch_id)

            logger.info(
                f"Batch {batch_id}: {status['status']} - "
                f"completed: {status['request_counts'].completed if status['request_counts'] else 0}, "
                f"failed: {status['request_counts'].failed if status['request_counts'] else 0}"
            )

            if status["status"] == "completed":
                # Handle case where all requests failed (no output file)
                if status["output_file_id"] is None:
                    # Check if there's an error file we can examine
                    if status.get("error_file_id"):
                        error_results = await self._download_error_file(status["error_file_id"])
                        # Return errors in a format the caller can handle
                        return error_results
                    else:
                        raise RuntimeError(
                            f"Batch {batch_id} completed but all requests failed "
                            f"(completed: {status['request_counts'].completed if status['request_counts'] else 0}, "
                            f"failed: {status['request_counts'].failed if status['request_counts'] else 0})"
                        )
                return await self._download_results(status["output_file_id"])

            elif status["status"] in ("failed", "expired", "cancelled"):
                error_msg = f"Batch {batch_id} {status['status']}"
                if status.get("error_file_id"):
                    errors = await self._download_results(status["error_file_id"])
                    error_msg += f": {errors}"
                raise RuntimeError(error_msg)

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")

            await asyncio.sleep(poll_interval)

    async def _download_results(self, file_id: str) -> Dict[str, str]:
        """Download and parse batch results."""
        content = await self.client.files.content(file_id)

        results = {}
        for line in content.text.strip().split('\n'):
            if not line:
                continue
            result = json.loads(line)
            custom_id = result["custom_id"]

            if result.get("response", {}).get("status_code") == 200:
                body = result["response"]["body"]
                content = body["choices"][0]["message"]["content"]
                results[custom_id] = content
            else:
                error = result.get("error", {})
                results[custom_id] = f"ERROR: {error.get('message', 'Unknown error')}"

        return results

    async def _download_error_file(self, file_id: str) -> Dict[str, str]:
        """Download and parse batch error file.

        Returns errors in the same format as _download_results so callers
        can handle them uniformly.
        """
        content = await self.client.files.content(file_id)

        results = {}
        first_error = None
        for line in content.text.strip().split('\n'):
            if not line:
                continue
            result = json.loads(line)
            custom_id = result["custom_id"]

            # Extract error info from the response
            response = result.get("response", {})
            error_body = response.get("body", {}).get("error", {})
            error_msg = error_body.get("message", "Unknown error")

            if first_error is None:
                first_error = error_msg
                logger.error(f"First batch error: {error_msg}")

            results[custom_id] = f"ERROR: {error_msg}"

        return results


class BatchBenchmarkRunner:
    """Run HaluMem benchmark using Batch API.

    This version submits all requests as batches, waits for completion,
    then computes metrics. Much more reliable for large benchmarks.
    """

    def __init__(self, client, config, dataset, output_dir: Path = None):
        self.client = client
        self.config = config
        self.dataset = dataset
        self.processor = BatchProcessor(client, output_dir)
        self.model = "gpt-5.1"
        self.small_model = "gpt-5-nano"

    async def run_extraction_batch(self) -> Dict[str, List[str]]:
        """Run extraction phase as a batch job.

        Returns:
            Dict mapping dialogue_id to list of extracted memories.
        """
        dialogues = self.dataset.get_dialogues()

        # Create extraction requests
        requests = []
        for i, dialogue in enumerate(dialogues):
            dialogue_text = "\n".join([
                f"{turn.role}: {turn.content}"
                for turn in dialogue
            ])

            prompt = """Extract important facts, preferences, and personal information from this conversation.
Return each memory as a separate line. Be concise - each memory should be one clear statement.
Focus on facts about the user that would be useful to remember for future conversations.

CONVERSATION:
""" + dialogue_text + """

MEMORIES (one per line):"""

            requests.append(self.processor._create_batch_request(
                custom_id=f"extract_{i}",
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            ))

        logger.info(f"Submitting extraction batch with {len(requests)} requests")
        batch = await self.processor.submit_batch(requests, "HaluMem extraction")

        logger.info(f"Waiting for extraction batch {batch.batch_id}...")
        results = await self.processor.wait_for_completion(batch.batch_id)

        # Parse results
        extracted = {}
        for custom_id, response in results.items():
            idx = int(custom_id.split("_")[1])
            memories = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line.startswith(('-', '*', '•')):
                    line = line[1:].strip()
                elif len(line) > 2 and line[0].isdigit() and line[1] in '.):':
                    line = line[2:].strip()
                if line and not line.startswith("ERROR"):
                    memories.append(line)
            extracted[f"dialogue_{idx}"] = memories

        return extracted

    async def run_update_batch(self) -> Dict[str, Dict[str, str]]:
        """Run update evaluation phase as a batch job.

        Returns:
            Dict mapping scenario_id to decision dict.
        """
        scenarios = self.dataset.get_update_scenarios()

        requests = []
        for scenario in scenarios:
            prompt = f"""You are a memory management system. Given an existing memory and new information, decide what action to take.

EXISTING MEMORY:
{scenario.old_memory}

NEW INFORMATION:
{scenario.new_information}

Decide:
- UPDATE: New info contradicts/refines the existing memory
- ADD: New info is additional, doesn't replace existing
- DELETE: Existing memory is now obsolete/wrong
- NOOP: No change needed (already captured or irrelevant)

Respond in exactly this format:
ACTION: <UPDATE|ADD|DELETE|NOOP>
RESULT: <new memory text if UPDATE/ADD, otherwise N/A>
REASONING: <brief explanation>"""

            requests.append(self.processor._create_batch_request(
                custom_id=f"update_{scenario.scenario_id}",
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            ))

        logger.info(f"Submitting update batch with {len(requests)} requests")
        batch = await self.processor.submit_batch(requests, "HaluMem updates")

        logger.info(f"Waiting for update batch {batch.batch_id}...")
        results = await self.processor.wait_for_completion(batch.batch_id)

        # Parse results
        decisions = {}
        for custom_id, response in results.items():
            scenario_id = custom_id.replace("update_", "")

            decision = {"action": "NOOP", "result": "", "reasoning": ""}
            for line in response.strip().split('\n'):
                if line.startswith("ACTION:"):
                    action = line.split(":", 1)[1].strip().upper()
                    if action in ("UPDATE", "ADD", "DELETE", "NOOP"):
                        decision["action"] = action
                elif line.startswith("RESULT:"):
                    decision["result"] = line.split(":", 1)[1].strip()
                elif line.startswith("REASONING:"):
                    decision["reasoning"] = line.split(":", 1)[1].strip()

            decisions[scenario_id] = decision

        return decisions

    async def run_qa_batch(
        self,
        memories: Dict[str, str],
        sample_size: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Run QA evaluation phase as batch jobs.

        This runs two batches:
        1. Answer generation (gpt-5.1)
        2. Answer judging (gpt-5.1)

        Note: Relevance scoring still uses real-time API with gpt-5-nano
        since it's cheap and we need the scores before generating answers.

        Returns:
            Dict mapping qa_id to result dict.
        """
        qa_pairs = self.dataset.get_qa_pairs(limit=sample_size)

        # For simplicity, we'll use all memories as context
        # In a real scenario, you'd do retrieval first
        memory_context = "\n".join([
            f"- {text}" for text in list(memories.values())[:50]  # Limit for context
        ])

        # Create answer generation requests
        answer_requests = []
        for qa in qa_pairs:
            prompt = f"""Based on the following memories, answer the question concisely.

MEMORIES:
{memory_context}

QUESTION: {qa.question}

ANSWER:"""

            answer_requests.append(self.processor._create_batch_request(
                custom_id=f"answer_{qa.qa_id}",
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            ))

        logger.info(f"Submitting QA answer batch with {len(answer_requests)} requests")
        batch = await self.processor.submit_batch(answer_requests, "HaluMem QA answers")

        logger.info(f"Waiting for QA answer batch {batch.batch_id}...")
        answers = await self.processor.wait_for_completion(batch.batch_id)

        # Create judging requests
        judge_requests = []
        for qa in qa_pairs:
            answer = answers.get(f"answer_{qa.qa_id}", "[No answer generated]")

            prompt = f"""You are evaluating an AI's answer to a question based on stored memories.

QUESTION: {qa.question}

GROUND TRUTH ANSWER: {qa.ground_truth_answer}

AI'S ANSWER: {answer}

Evaluate the AI's answer:
1. Is it factually correct based on the ground truth? (0.0 to 1.0)
2. Does it contain hallucinated (made up) information not in ground truth?
3. Does it omit important information from the ground truth?

Respond in exactly this format:
CORRECTNESS: <0.0 to 1.0>
HALLUCINATION: <YES|NO>
OMISSION: <YES|NO>
REASONING: <brief explanation>"""

            judge_requests.append(self.processor._create_batch_request(
                custom_id=f"judge_{qa.qa_id}",
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            ))

        logger.info(f"Submitting QA judge batch with {len(judge_requests)} requests")
        batch = await self.processor.submit_batch(judge_requests, "HaluMem QA judging")

        logger.info(f"Waiting for QA judge batch {batch.batch_id}...")
        judgments = await self.processor.wait_for_completion(batch.batch_id)

        # Combine results
        results = {}
        for qa in qa_pairs:
            answer = answers.get(f"answer_{qa.qa_id}", "[Error]")
            judgment_text = judgments.get(f"judge_{qa.qa_id}", "")

            # Parse judgment
            correctness = 0.0
            hallucination = False
            omission = False
            reasoning = ""

            for line in judgment_text.strip().split('\n'):
                if line.startswith("CORRECTNESS:"):
                    try:
                        correctness = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("HALLUCINATION:"):
                    hallucination = "YES" in line.upper()
                elif line.startswith("OMISSION:"):
                    omission = "YES" in line.upper()
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            results[qa.qa_id] = {
                "question": qa.question,
                "answer": answer,
                "ground_truth": qa.ground_truth_answer,
                "correctness": correctness,
                "hallucination": hallucination,
                "omission": omission,
                "reasoning": reasoning
            }

        return results
