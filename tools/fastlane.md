# Fastlane

## Description

Fastlane automates App Store Connect tasks. This recipe covers `deliver`, the part of fastlane that downloads and uploads the App Store metadata (name, subtitle, keywords, description, etc.) and screenshots.

## Installation

Install fastlane with Homebrew (it comes with its own Ruby):

```
$ brew install fastlane
```

To update it:

```
$ brew upgrade fastlane
```

Fastlane prints a warning if the locale is not UTF-8. Add these lines to `~/.zshrc`:

```
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

## Set up an existing repository

Go to the folder that contains the `.xcodeproj` and run:

```
$ fastlane init
```

When asked what you want to use fastlane for, pick **4 (Manual setup)**. The other choices sign in with your Apple ID and write lanes you may not want. This creates a `fastlane` folder with an `Appfile` and a `Fastfile`.

### Appfile

The `Appfile` holds the bundle identifier of the app:

```
app_identifier("com.example.myapp")
```

### Deliverfile

Create `fastlane/Deliverfile`. It tells `deliver` how to sign in to App Store Connect. There are two options.

**Option 1: API key (recommended).** No password and no 2FA prompt.

```
api_key_path("/path/outside/the/repo/api_key.json")
precheck_include_in_app_purchases(false)
```

The second line is needed with an API key: precheck cannot check in-app purchases when signed in with a key.

The JSON file looks like this (`key` is the content of the `.p8` file, with `\n` for line breaks):

```
{
  "key_id": "ABC123DEFG",
  "issuer_id": "00000000-0000-0000-0000-000000000000",
  "key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----",
  "in_house": false
}
```

To create a key: App Store Connect > Users and Access > Integrations > App Store Connect API > Team Keys. Notes:

- Only the Account Holder or an Admin can see this page and create a key.
- A key only works inside the account where it was created. A key from a personal account cannot see the apps of a company account.
- Pick the **App Manager** role to read and edit metadata.
- The `.p8` file can be downloaded only once.
- Never put the key inside the repository.

**Option 2: Apple ID.** Fastlane signs in as you, with the access you have in App Store Connect.

```
username("me@example.com")
```

Fastlane asks for the password and a 2FA code, then keeps the session for about a month. If the Apple ID belongs to several teams, it asks which one to use (add `itc_team_id("...")` to the `Appfile` to skip the question).

If the sign-in fails with error `-900007`, Apple refused it. Sign in at appstoreconnect.apple.com in a browser to check the account (new agreement to accept, locked account, etc.), then delete the password fastlane saved and try again:

```
$ fastlane fastlane-credentials remove --username me@example.com
```

If it still fails, use an API key.

### .gitignore

Add these lines to the `.gitignore` (adjust the path if the `fastlane` folder is not at the root):

```
fastlane/report.xml
fastlane/Preview.html
fastlane/test_output
```

## Download metadata and screenshots

Run these commands from the folder that contains the `fastlane` folder.

To download the metadata into `fastlane/metadata` (one folder per language, one text file per field):

```
$ fastlane deliver download_metadata
```

To download the screenshots into `fastlane/screenshots` (one folder per language):

```
$ fastlane deliver download_screenshots
```

By default, `deliver` downloads from the version being edited ("Prepare for Submission"). If there is none, it fails with `Could not find an edit version on App Store Connect`. Add `--use_live_version true` to download from the version live on the App Store:

```
$ fastlane deliver download_metadata --use_live_version true
$ fastlane deliver download_screenshots --use_live_version true
```

## Upload metadata and screenshots

An upload needs a version open for editing on App Store Connect. Pass `--app_version` and `deliver` creates it if it does not exist:

```
$ fastlane deliver --app_version "1.2.0" ...
```

`--force` skips the HTML preview that asks for confirmation. `--skip_binary_upload` tells `deliver` not to look for an `.ipa`. Nothing is submitted for review unless you add `--submit_for_review true`.

To upload the metadata only:

```
$ fastlane deliver --skip_binary_upload --skip_screenshots --force
```

The text fields on App Store Connect are replaced by the content of the files. A field with no file is left as it is.

To upload the screenshots only, deleting the ones already on App Store Connect first:

```
$ fastlane deliver --skip_binary_upload --skip_metadata --overwrite_screenshots --force
```

Without `--overwrite_screenshots`, the new screenshots are added next to the existing ones.

To upload both:

```
$ fastlane deliver --skip_binary_upload --overwrite_screenshots --force
```
