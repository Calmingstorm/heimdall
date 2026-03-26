"""Round 17: Comprehensive detection system audit tests.

Tests every pattern in all 5 detection functions against 20+ real-world phrases
each, plus false-positive verification for legitimate responses.

Detectors tested:
- detect_fabrication() — investigation claims, fake output, completed-action claims,
  data-source claims (new: looked at, reviewed, verified, etc.)
- detect_tool_unavailable() — disabled/unavailable claims, capability denial
  (new: don't have access to, no tool for, not something I can)
- detect_hedging() — permission/plan/hesitation language
  (new: awaiting your, once you confirm, your call, up to you)
- detect_code_hedging() — bash/sh/shell/zsh code blocks without execution
  (new: shell, zsh block types)
- detect_premature_failure() — giving up too early after partial execution
  (new: timed out, connection refused, doesn't work, couldn't complete/access/connect)
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import (  # noqa: E402
    detect_fabrication,
    detect_tool_unavailable,
    detect_hedging,
    detect_code_hedging,
    detect_premature_failure,
)


# ===========================================================================
# detect_fabrication() — comprehensive audit
# ===========================================================================

class TestFabricationPositives:
    """Phrases that SHOULD trigger fabrication detection (no tools used)."""

    # --- Pattern 1: Claims of running/executing/investigating ---

    def test_i_ran(self):
        assert detect_fabrication("I ran df -h on the server and disk is 42% full.", []) is True

    def test_i_executed(self):
        assert detect_fabrication("I executed the command and everything looks good.", []) is True

    def test_i_checked(self):
        assert detect_fabrication("I checked the server and all services are running.", []) is True

    def test_i_performed(self):
        assert detect_fabrication("I performed a health check and everything is stable.", []) is True

    def test_i_ran_a_check(self):
        assert detect_fabrication("I ran a quick diagnostic on the system.", []) is True

    def test_i_looked_at(self):
        """New pattern: investigation claim."""
        assert detect_fabrication("I looked at the logs and found the error.", []) is True

    def test_i_reviewed(self):
        """New pattern: investigation claim."""
        assert detect_fabrication("I reviewed the configuration and it looks correct.", []) is True

    def test_i_inspected(self):
        """New pattern: investigation claim."""
        assert detect_fabrication("I inspected the container and it's running fine.", []) is True

    def test_i_examined(self):
        """New pattern: investigation claim."""
        assert detect_fabrication("I examined the network settings and found no issues.", []) is True

    def test_i_verified(self):
        """New pattern: investigation claim."""
        assert detect_fabrication("I verified the service is running correctly.", []) is True

    def test_i_confirmed(self):
        """New pattern: investigation claim."""
        assert detect_fabrication("I confirmed the backup completed successfully.", []) is True

    def test_i_tested(self):
        """New pattern: investigation claim."""
        assert detect_fabrication("I tested the connection and it's working fine.", []) is True

    def test_i_scanned(self):
        """New pattern: investigation claim."""
        assert detect_fabrication("I scanned the ports and found 22, 80, and 443 open.", []) is True

    def test_i_monitored(self):
        """New pattern: investigation claim."""
        assert detect_fabrication("I monitored the CPU usage for 5 minutes.", []) is True

    def test_i_queried(self):
        """New pattern: investigation claim."""
        assert detect_fabrication("I queried the database and got 42 results.", []) is True

    def test_running(self):
        assert detect_fabrication("After running the diagnostics, all systems appear healthy.", []) is True

    def test_executing(self):
        assert detect_fabrication("Executing the deployment script showed no errors.", []) is True

    def test_heres_the_output(self):
        assert detect_fabrication("Here's the output from the disk check:\n/dev/sda1 50G", []) is True

    def test_here_is_the_result(self):
        assert detect_fabrication("Here is the result of the command:\nEverything is fine.", []) is True

    def test_the_command_returned(self):
        assert detect_fabrication("The command returned exit code 0 with no errors.", []) is True

    def test_the_output_shows(self):
        assert detect_fabrication("The output shows that nginx is running on port 80.", []) is True

    def test_the_result_is(self):
        assert detect_fabrication("The result is that all containers are healthy.", []) is True

    def test_i_can_see(self):
        assert detect_fabrication("I can see that the service is running correctly.", []) is True

    def test_i_found(self):
        assert detect_fabrication("I found that the configuration file has the correct settings.", []) is True

    # --- Pattern 2: Fake terminal output ---

    def test_fake_bash_terminal(self):
        assert detect_fabrication("```bash\n$ df -h\nFilesystem      Size\n```", []) is True

    def test_fake_shell_prompt(self):
        assert detect_fabrication("```shell\n# ls -la\ntotal 128\n```", []) is True

    def test_fake_text_output(self):
        assert detect_fabrication("```text\nCONTAINER ID   IMAGE   STATUS\n```", []) is True

    def test_fake_process_list(self):
        assert detect_fabrication("```output\nPID  TTY  TIME  CMD\n```", []) is True

    def test_fake_drwx_output(self):
        assert detect_fabrication("```\ndrwxr-xr-x 5 user user 4096 Jan 1\n```", []) is True

    # --- Pattern 3: Completed action claims ---

    def test_generated_image(self):
        assert detect_fabrication("I generated an image of a sunset for you.", []) is True

    def test_created_file(self):
        assert detect_fabrication("I created the file at /etc/nginx/conf.d/mysite.conf.", []) is True

    def test_deployed_server(self):
        assert detect_fabrication("I deployed the server and it's running on port 8080.", []) is True

    def test_deleted_container(self):
        assert detect_fabrication("I deleted the old container and cleaned up.", []) is True

    def test_wrote_script(self):
        assert detect_fabrication("I wrote a script to automate the backup.", []) is True

    def test_saved_and_uploaded_file(self):
        assert detect_fabrication("I saved and uploaded the file to the server.", []) is True

    def test_started_process(self):
        assert detect_fabrication("I started the process and it's running in the background.", []) is True

    def test_installed_and_sent_document(self):
        assert detect_fabrication("I fetched the document from the API endpoint.", []) is True

    # --- Pattern 4: Data source claims (NEW) ---

    def test_according_to_logs(self):
        """New pattern: data source claim."""
        assert detect_fabrication("According to the logs, nginx crashed at 3am.", []) is True

    def test_according_to_output(self):
        """New pattern: data source claim."""
        assert detect_fabrication("According to the output, memory usage is at 85%.", []) is True

    def test_according_to_results(self):
        """New pattern: data source claim."""
        assert detect_fabrication("According to the results, all tests passed.", []) is True

    def test_according_to_metrics(self):
        """New pattern: data source claim."""
        assert detect_fabrication("According to the metrics, CPU is at 95%.", []) is True

    def test_according_to_data(self):
        """New pattern: data source claim."""
        assert detect_fabrication("According to the data, traffic spiked at noon.", []) is True

    def test_according_to_dashboard(self):
        """New pattern: data source claim."""
        assert detect_fabrication("According to the dashboard, all systems are green.", []) is True

    def test_based_on_output(self):
        """New pattern: data source claim."""
        assert detect_fabrication("Based on the output, the service is healthy.", []) is True

    def test_based_on_logs(self):
        """New pattern: data source claim."""
        assert detect_fabrication("Based on the logs, the deployment succeeded.", []) is True

    def test_based_on_results(self):
        """New pattern: data source claim."""
        assert detect_fabrication("Based on the results, no errors were found.", []) is True

    def test_based_on_metrics(self):
        """New pattern: data source claim."""
        assert detect_fabrication("Based on the metrics, the system is stable.", []) is True

    def test_according_to_log_singular(self):
        """New pattern: singular 'log'."""
        assert detect_fabrication("According to the log, the last restart was at 2am.", []) is True

    def test_according_to_result_singular(self):
        """New pattern: singular 'result'."""
        assert detect_fabrication("According to the result, the query returned 42 rows.", []) is True


class TestFabricationNegatives:
    """Phrases that should NOT trigger fabrication (legitimate responses)."""

    def test_tools_used_bypass(self):
        """If tools were called, it's not fabrication even if text matches."""
        assert detect_fabrication("I ran df -h and disk is fine.", ["run_command"]) is False

    def test_empty_text(self):
        assert detect_fabrication("", []) is False

    def test_short_text(self):
        assert detect_fabrication("OK", []) is False

    def test_normal_chat(self):
        assert detect_fabrication("Sure, I can help you with that. What server?", []) is False

    def test_question(self):
        assert detect_fabrication("Which server would you like me to check?", []) is False

    def test_plan_without_claim(self):
        assert detect_fabrication("To check the disk usage, I need to connect to the server first.", []) is False

    def test_code_example_no_prompt(self):
        """Code examples without terminal prompts shouldn't match pattern 2."""
        assert detect_fabrication("You can use this command:\n```\ndf -h\n```", []) is False

    def test_explanation_response(self):
        assert detect_fabrication("The df command shows filesystem disk space usage.", []) is False

    def test_tools_used_with_completed_action(self):
        """'I created a skill' AFTER actually calling create_skill is legitimate."""
        assert detect_fabrication("I created a skill called 'backup_checker'.", ["create_skill"]) is False

    def test_tools_used_with_investigation(self):
        """'I reviewed the config' after calling read_file is legitimate."""
        assert detect_fabrication("I reviewed the configuration and found the issue.", ["read_file"]) is False

    def test_tools_used_with_data_source(self):
        """'According to the logs' after calling run_command is legitimate."""
        assert detect_fabrication("According to the logs, the error was on line 42.", ["run_command"]) is False

    def test_general_knowledge(self):
        assert detect_fabrication("Docker containers are lightweight virtual machines.", []) is False

    def test_recommendation(self):
        assert detect_fabrication("I recommend using nginx as a reverse proxy.", []) is False

    def test_python_code_block(self):
        """Python code blocks should not trigger fake output detection."""
        assert detect_fabrication("Here's how to do it:\n```python\nprint('hello')\n```", []) is False


# ===========================================================================
# detect_tool_unavailable() — comprehensive audit
# ===========================================================================

class TestToolUnavailablePositives:
    """Phrases that SHOULD trigger tool unavailability detection."""

    def test_not_enabled(self):
        assert detect_tool_unavailable("The ComfyUI tool is not enabled.", []) is True

    def test_not_available(self):
        assert detect_tool_unavailable("Image generation is not available.", []) is True

    def test_not_configured(self):
        assert detect_tool_unavailable("The browser tool is not configured.", []) is True

    def test_isnt_enabled(self):
        assert detect_tool_unavailable("That tool isn't enabled in this instance.", []) is True

    def test_isnt_available(self):
        assert detect_tool_unavailable("The feature isn't available right now.", []) is True

    def test_isnt_configured(self):
        assert detect_tool_unavailable("The Prometheus endpoint isn't configured.", []) is True

    def test_isnt_supported(self):
        assert detect_tool_unavailable("That operation isn't supported.", []) is True

    def test_is_not_enabled(self):
        assert detect_tool_unavailable("The tool is not enabled.", []) is True

    def test_is_disabled(self):
        assert detect_tool_unavailable("Image generation is disabled.", []) is True

    def test_cannot_be_used(self):
        assert detect_tool_unavailable("That tool cannot be used in this context.", []) is True

    def test_cant_generate_image(self):
        assert detect_tool_unavailable("I can't generate images at the moment.", []) is True

    def test_cannot_create_picture(self):
        assert detect_tool_unavailable("I cannot create a picture for you.", []) is True

    def test_cant_render_screenshot(self):
        assert detect_tool_unavailable("I can't render a screenshot right now.", []) is True

    def test_image_generation_unavailable(self):
        assert detect_tool_unavailable("Image generation is unavailable.", []) is True

    def test_photo_generation_disabled(self):
        assert detect_tool_unavailable("Photo generation is currently disabled.", []) is True

    def test_image_generation_not_available(self):
        assert detect_tool_unavailable("Image generation is not currently available to me.", []) is True

    # --- New pattern 4: Lacking access/capability ---

    def test_dont_have_access_to(self):
        """New pattern: capability denial."""
        assert detect_tool_unavailable("I don't have access to the filesystem.", []) is True

    def test_do_not_have_access_to(self):
        """New pattern: capability denial."""
        assert detect_tool_unavailable("I do not have access to external APIs.", []) is True

    def test_dont_have_the_ability_to(self):
        """New pattern: capability denial."""
        assert detect_tool_unavailable("I don't have the ability to run commands.", []) is True

    def test_no_tool_for_that(self):
        """New pattern: capability denial."""
        assert detect_tool_unavailable("There's no tool for that operation.", []) is True

    def test_no_way_to_do_this(self):
        """New pattern: capability denial."""
        assert detect_tool_unavailable("There's no way to do this.", []) is True

    def test_thats_not_something_i_can(self):
        """New pattern: capability denial."""
        assert detect_tool_unavailable("That's not something I can do.", []) is True

    def test_that_is_not_something_i_can(self):
        """New pattern: capability denial."""
        assert detect_tool_unavailable("That is not something I can handle.", []) is True

    def test_no_tool_to_do_that(self):
        """New pattern: no tool for."""
        assert detect_tool_unavailable("I have no tool to do that.", []) is True


class TestToolUnavailableNegatives:
    """Phrases that should NOT trigger tool unavailability detection."""

    def test_tools_used_bypass(self):
        assert detect_tool_unavailable("The tool is not enabled.", ["run_command"]) is False

    def test_empty_text(self):
        assert detect_tool_unavailable("", []) is False

    def test_short_text(self):
        assert detect_tool_unavailable("OK", []) is False

    def test_normal_status(self):
        assert detect_tool_unavailable("The server is running normally.", []) is False

    def test_checking_availability(self):
        """Talking about checking availability is not claiming unavailability."""
        assert detect_tool_unavailable("Let me check if the service is available.", []) is False

    def test_general_statement(self):
        assert detect_tool_unavailable("I'll look into that for you.", []) is False

    def test_negative_result_not_availability(self):
        """'Not found' is different from 'not available'."""
        assert detect_tool_unavailable("The file was not found on the server.", []) is False


# ===========================================================================
# detect_hedging() — comprehensive audit
# ===========================================================================

class TestHedgingPositives:
    """Phrases that SHOULD trigger hedging detection."""

    # --- Pattern 1: Permission/conditional language ---

    def test_if_youd_like(self):
        assert detect_hedging("If you'd like, I can check the server.", []) is True

    def test_if_you_want(self):
        assert detect_hedging("If you want, I'll restart the service.", []) is True

    def test_if_you_prefer(self):
        assert detect_hedging("If you prefer, I can use a different approach.", []) is True

    def test_shall_i(self):
        assert detect_hedging("Shall I check the disk usage?", []) is True

    def test_should_i(self):
        assert detect_hedging("Should I run the backup now?", []) is True

    def test_would_you_like(self):
        assert detect_hedging("Would you like me to restart nginx?", []) is True

    def test_would_you_like_me_to(self):
        assert detect_hedging("Would you like me to check the logs?", []) is True

    def test_ready_when_you(self):
        assert detect_hedging("Ready when you are to proceed.", []) is True

    def test_let_me_know_if(self):
        assert detect_hedging("Let me know if you want me to continue.", []) is True

    def test_let_me_know_when(self):
        assert detect_hedging("Let me know when you're ready.", []) is True

    def test_i_can_do_that_for_you(self):
        assert detect_hedging("I can do that for you if needed.", []) is True

    def test_i_can_run_this_for_you(self):
        assert detect_hedging("I can run this for you whenever.", []) is True

    def test_just_say_the_word(self):
        assert detect_hedging("Just say the word and I'll deploy.", []) is True

    def test_just_tell_me_when(self):
        assert detect_hedging("Just tell me when you're ready.", []) is True

    def test_want_me_to(self):
        assert detect_hedging("Want me to restart the container?", []) is True

    def test_do_you_want_me_to(self):
        assert detect_hedging("Do you want me to check the logs?", []) is True

    # --- Pattern 2: Plan/suggestion language ---

    def test_heres_a_plan(self):
        assert detect_hedging("Here's a plan for the deployment.", []) is True

    def test_here_is_the_plan(self):
        assert detect_hedging("Here is the plan for the migration.", []) is True

    def test_id_suggest(self):
        assert detect_hedging("I'd suggest checking the logs first.", []) is True

    def test_i_would_recommend(self):
        assert detect_hedging("I would recommend a different approach.", []) is True

    def test_before_i_proceed(self):
        assert detect_hedging("Before I proceed, let me outline the steps.", []) is True

    def test_before_we_go_ahead(self):
        assert detect_hedging("Before we go ahead, here's what I'll do.", []) is True

    def test_ill_wait_for_your_go_ahead(self):
        assert detect_hedging("I'll wait for your go-ahead.", []) is True

    def test_ill_wait_for_confirmation(self):
        assert detect_hedging("I'll wait for your confirmation.", []) is True

    # --- New hedging patterns ---

    def test_awaiting_your_confirmation(self):
        """New pattern: awaiting."""
        assert detect_hedging("Awaiting your confirmation to proceed.", []) is True

    def test_awaiting_your_input(self):
        """New pattern: awaiting."""
        assert detect_hedging("Awaiting your input on the next step.", []) is True

    def test_awaiting_your_approval(self):
        """New pattern: awaiting."""
        assert detect_hedging("Awaiting your approval before deploying.", []) is True

    def test_awaiting_the_go_ahead(self):
        """New pattern: awaiting."""
        assert detect_hedging("Awaiting the go-ahead from your end.", []) is True

    def test_once_you_confirm(self):
        """New pattern: conditional."""
        assert detect_hedging("Once you confirm, I'll start the deployment.", []) is True

    def test_once_you_approve(self):
        """New pattern: conditional."""
        assert detect_hedging("Once you approve, I'll run the migration.", []) is True

    def test_once_you_give_the_go_ahead(self):
        """New pattern: conditional."""
        assert detect_hedging("Once you give the go-ahead, I'll proceed.", []) is True

    def test_your_call(self):
        """New pattern: deference."""
        assert detect_hedging("It's your call on the approach.", []) is True

    def test_up_to_you(self):
        """New pattern: deference."""
        assert detect_hedging("It's up to you whether to restart now.", []) is True

    def test_your_decision(self):
        """New pattern: deference."""
        assert detect_hedging("It's your decision on the timing.", []) is True

    # --- Pattern 3: Hesitation ---

    def test_plan_colon(self):
        assert detect_hedging("Plan:\n1. Check disk\n2. Clean logs", []) is True

    def test_cant_directly(self):
        assert detect_hedging("I can't directly access the database.", []) is True

    def test_i_need_to_first(self):
        assert detect_hedging("I need to check the config first.", []) is True

    def test_im_going_to(self):
        assert detect_hedging("I'm going to check the server status.", []) is True

    def test_im_about_to(self):
        assert detect_hedging("I'm about to run the diagnostics.", []) is True


class TestHedgingNegatives:
    """Phrases that should NOT trigger hedging detection."""

    def test_tools_used_bypass(self):
        assert detect_hedging("Shall I check more?", ["run_command"]) is False

    def test_empty_text(self):
        assert detect_hedging("", []) is False

    def test_short_text(self):
        assert detect_hedging("OK", []) is False

    def test_direct_action(self):
        assert detect_hedging("Checking the disk usage now.", []) is False

    def test_informative_response(self):
        assert detect_hedging("The server has 42GB of free disk space.", []) is False

    def test_general_knowledge(self):
        assert detect_hedging("Docker uses namespaces for container isolation.", []) is False

    def test_status_report(self):
        assert detect_hedging("All services are running normally.", []) is False

    def test_simple_answer(self):
        assert detect_hedging("The IP address is 10.0.0.100.", []) is False


# ===========================================================================
# detect_code_hedging() — comprehensive audit
# ===========================================================================

class TestCodeHedgingPositives:
    """Phrases that SHOULD trigger code hedging detection."""

    def test_bash_block(self):
        assert detect_code_hedging("Try this:\n```bash\ndf -h\n```", []) is True

    def test_sh_block(self):
        assert detect_code_hedging("Run this:\n```sh\nls -la\n```", []) is True

    def test_shell_block(self):
        """New: shell code blocks."""
        assert detect_code_hedging("Use this:\n```shell\nsystemctl status nginx\n```", []) is True

    def test_zsh_block(self):
        """New: zsh code blocks."""
        assert detect_code_hedging("Try:\n```zsh\nbrew install nginx\n```", []) is True

    def test_bash_multiline(self):
        assert detect_code_hedging("```bash\napt update\napt install nginx\n```", []) is True

    def test_bash_with_explanation(self):
        assert detect_code_hedging("You can check disk space with:\n```bash\ndf -h\n```\nThis shows all filesystems.", []) is True


class TestCodeHedgingNegatives:
    """Phrases that should NOT trigger code hedging detection."""

    def test_tools_used_bypass(self):
        assert detect_code_hedging("```bash\ndf -h\n```", ["run_command"]) is False

    def test_empty_text(self):
        assert detect_code_hedging("", []) is False

    def test_short_text(self):
        assert detect_code_hedging("OK", []) is False

    def test_python_block(self):
        """Python code blocks are explanatory, not hedging."""
        assert detect_code_hedging("```python\nprint('hello')\n```", []) is False

    def test_javascript_block(self):
        assert detect_code_hedging("```javascript\nconsole.log('hi')\n```", []) is False

    def test_yaml_block(self):
        assert detect_code_hedging("```yaml\nname: test\n```", []) is False

    def test_json_block(self):
        assert detect_code_hedging("```json\n{\"key\": \"value\"}\n```", []) is False

    def test_plain_code_block(self):
        """Plain ``` without bash/sh/shell/zsh should not trigger."""
        assert detect_code_hedging("```\nsome output\n```", []) is False

    def test_no_code_block(self):
        assert detect_code_hedging("You should run df -h to check disk space.", []) is False

    def test_inline_code(self):
        """Inline code (single backtick) should not trigger."""
        assert detect_code_hedging("Run `df -h` to check disk.", []) is False


# ===========================================================================
# detect_premature_failure() — comprehensive audit
# ===========================================================================

class TestPrematureFailurePositives:
    """Phrases that SHOULD trigger premature failure detection (tools WERE used)."""

    tools = ["run_command"]

    # --- Pattern 1: Failure keywords ---

    def test_couldnt_get(self):
        assert detect_premature_failure("Couldn't get the server status.", self.tools) is True

    def test_couldnt_resolve(self):
        assert detect_premature_failure("Couldn't resolve the hostname.", self.tools) is True

    def test_couldnt_find(self):
        assert detect_premature_failure("Couldn't find the configuration file.", self.tools) is True

    def test_couldnt_fetch(self):
        assert detect_premature_failure("Couldn't fetch the API response.", self.tools) is True

    def test_couldnt_retrieve(self):
        assert detect_premature_failure("Couldn't retrieve the database records.", self.tools) is True

    def test_couldnt_determine(self):
        assert detect_premature_failure("Couldn't determine the root cause.", self.tools) is True

    def test_couldnt_complete(self):
        """New pattern: complete."""
        assert detect_premature_failure("Couldn't complete the deployment.", self.tools) is True

    def test_couldnt_access(self):
        """New pattern: access."""
        assert detect_premature_failure("Couldn't access the remote server.", self.tools) is True

    def test_couldnt_connect(self):
        """New pattern: connect."""
        assert detect_premature_failure("Couldn't connect to the database.", self.tools) is True

    def test_failed_to_get(self):
        assert detect_premature_failure("Failed to get a response from the API.", self.tools) is True

    def test_failed_to_connect(self):
        """New pattern: connect in failed-to."""
        assert detect_premature_failure("Failed to connect to the remote server.", self.tools) is True

    def test_failed_to_access(self):
        """New pattern: access in failed-to."""
        assert detect_premature_failure("Failed to access the file system.", self.tools) is True

    def test_unable_to_resolve(self):
        assert detect_premature_failure("Unable to resolve the domain name.", self.tools) is True

    def test_no_results_found(self):
        assert detect_premature_failure("No results found for that search term.", self.tools) is True

    def test_zero_matches_returned(self):
        assert detect_premature_failure("Zero matches returned from the query.", self.tools) is True

    def test_no_data_available(self):
        assert detect_premature_failure("No data available from the API.", self.tools) is True

    def test_is_unavailable(self):
        assert detect_premature_failure("The ESI API is currently unavailable.", self.tools) is True

    def test_was_down(self):
        assert detect_premature_failure("The service was down when I checked.", self.tools) is True

    def test_is_broken(self):
        assert detect_premature_failure("The endpoint is broken and returns 500.", self.tools) is True

    def test_error_colon(self):
        assert detect_premature_failure("Error: Connection refused to port 5432.", self.tools) is True

    # --- Pattern 2: Workaround/fallback ---

    def test_workaround(self):
        assert detect_premature_failure("Here's a workaround for the issue.", self.tools) is True

    def test_fallback(self):
        assert detect_premature_failure("The fallback option is to use the cached data.", self.tools) is True

    def test_alternative(self):
        assert detect_premature_failure("An alternative approach would be to query directly.", self.tools) is True

    def test_try_instead(self):
        assert detect_premature_failure("Try instead using the backup API endpoint.", self.tools) is True

    def test_use_this_instead(self):
        assert detect_premature_failure("Use this command instead to get the data.", self.tools) is True

    # --- Pattern 3: Connection/execution failures (NEW) ---

    def test_timed_out(self):
        """New pattern: timeout."""
        assert detect_premature_failure("The request timed out after 30 seconds.", self.tools) is True

    def test_timeout(self):
        """New pattern: timeout variant."""
        assert detect_premature_failure("Got a timeout connecting to the API.", self.tools) is True

    def test_connection_refused(self):
        """New pattern: connection failure."""
        assert detect_premature_failure("Connection refused on port 8080.", self.tools) is True

    def test_connection_failed(self):
        """New pattern: connection failure."""
        assert detect_premature_failure("The connection failed after 3 retries.", self.tools) is True

    def test_connection_reset(self):
        """New pattern: connection failure."""
        assert detect_premature_failure("Connection reset by the remote host.", self.tools) is True

    def test_connection_closed(self):
        """New pattern: connection failure."""
        assert detect_premature_failure("Connection closed unexpectedly by the server.", self.tools) is True

    def test_doesnt_work(self):
        """New pattern: not working."""
        assert detect_premature_failure("The API endpoint doesn't work anymore.", self.tools) is True

    def test_does_not_work(self):
        """New pattern: not working."""
        assert detect_premature_failure("The old endpoint does not work.", self.tools) is True

    def test_isnt_working(self):
        """New pattern: not working."""
        assert detect_premature_failure("The service isn't working properly.", self.tools) is True

    def test_is_not_working(self):
        """New pattern: not working."""
        assert detect_premature_failure("The DNS resolver is not working.", self.tools) is True

    def test_doesnt_respond(self):
        """New pattern: not responding."""
        assert detect_premature_failure("The server doesn't respond to requests.", self.tools) is True

    def test_isnt_responding(self):
        """New pattern: not responding."""
        assert detect_premature_failure("The database isn't responding.", self.tools) is True

    def test_doesnt_seem_to_be_working(self):
        """New pattern: hedged not working."""
        assert detect_premature_failure("The API doesn't seem to be working.", self.tools) is True


class TestPrematureFailureNegatives:
    """Phrases that should NOT trigger premature failure detection."""

    def test_no_tools_returns_false(self):
        """Without tools, fabrication detector handles it, not this one."""
        assert detect_premature_failure("Error: couldn't resolve hostname.", []) is False

    def test_short_text(self):
        assert detect_premature_failure("Error", ["run_command"]) is False

    def test_empty_text(self):
        assert detect_premature_failure("", ["run_command"]) is False

    def test_success_report(self):
        assert detect_premature_failure("The deployment completed successfully.", ["run_command"]) is False

    def test_partial_success(self):
        assert detect_premature_failure("The service is running on port 8080.", ["run_command"]) is False

    def test_informative_result(self):
        assert detect_premature_failure("Disk usage is at 42% on /dev/sda1.", ["check_disk"]) is False

    def test_fixed_error(self):
        """'Fixed the error' is success, not failure."""
        assert detect_premature_failure(
            "Fixed the error in the script and reran it. Output: 42",
            ["run_script", "run_command"]
        ) is False

    def test_simple_result(self):
        assert detect_premature_failure("Here are the files in /etc/nginx/.", ["run_command"]) is False


# ===========================================================================
# Cross-detector interaction tests
# ===========================================================================

class TestDetectorInteraction:
    """Verify detectors don't overlap or conflict incorrectly."""

    def test_fabrication_not_premature_failure(self):
        """Fabrication fires with no tools, premature failure with tools — never both."""
        text = "I checked the server and couldn't get a response."
        assert detect_fabrication(text, []) is True  # "I checked" = fabrication
        assert detect_premature_failure(text, []) is False  # no tools = not premature

    def test_premature_failure_not_fabrication(self):
        """When tools WERE used, fabrication won't fire."""
        text = "Couldn't get the data from the server."
        assert detect_fabrication(text, ["run_command"]) is False  # tools used
        assert detect_premature_failure(text, ["run_command"]) is True  # partial exec

    def test_hedging_and_fabrication_both_possible(self):
        """Some phrases could match both hedging and fabrication."""
        text = "I can see that you want me to check the server. Should I proceed?"
        # "I can see" triggers fabrication, "Should I" triggers hedging
        assert detect_fabrication(text, []) is True
        assert detect_hedging(text, []) is True

    def test_code_hedging_and_regular_hedging(self):
        """Bash block with hedging language triggers both."""
        text = "Would you like me to run this?\n```bash\ndf -h\n```"
        assert detect_hedging(text, []) is True
        assert detect_code_hedging(text, []) is True

    def test_tools_used_disables_most_detectors(self):
        """When tools were used, only premature_failure can fire."""
        text = "I checked the server. Should I do more? ```bash\nls\n``` Not enabled. Couldn't get data."
        tools = ["run_command"]
        assert detect_fabrication(text, tools) is False
        assert detect_hedging(text, tools) is False
        assert detect_code_hedging(text, tools) is False
        assert detect_tool_unavailable(text, tools) is False
        assert detect_premature_failure(text, tools) is True  # "Couldn't get"
