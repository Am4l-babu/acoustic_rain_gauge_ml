code = '''\
import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime
from difflib import SequenceMatcher

CSV_FILE = \'papers_database.csv\'
LINKS_TXT = r\'D:\\ARG_Research\\research papers\\research links new.txt\'

if not os.path.exists(CSV_FILE):
    df_init = pd.DataFrame(columns=[\'Name\', \'DOI\', \'Link\', \'Abstract\', \'Status\', \'Type\', \'Date_Added\'])
    df_init.to_csv(CSV_FILE, index=False)

def load_data():
    return pd.read_csv(CSV_FILE)

def save_data(df):
    df.to_csv(CSV_FILE, index=False)

def calculate_similarity(a, b):
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

# ── Text File Helpers ─────────────────────────────────────────────────────────

def extract_urls_from_txt():
    if not os.path.exists(LINKS_TXT):
        return set()
    url_pattern = re.compile(r\'https?://[^\\s\\r\\n]+\')
    urls = set()
    with open(LINKS_TXT, \'r\', encoding=\'utf-8\', errors=\'ignore\') as f:
        for line in f:
            for u in url_pattern.findall(line):
                urls.add(u.strip().rstrip(\'/\').lower())
    return urls

def is_link_in_txt(link, url_set=None):
    if not link:
        return False
    if url_set is None:
        url_set = extract_urls_from_txt()
    return link.strip().rstrip(\'/\').lower() in url_set

def count_entries_in_txt():
    if not os.path.exists(LINKS_TXT):
        return 0
    pattern = re.compile(r\'^\\s*(\\d+)\\.\')
    max_num = 0
    with open(LINKS_TXT, \'r\', encoding=\'utf-8\', errors=\'ignore\') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                n = int(m.group(1))
                if n > max_num:
                    max_num = n
    return max_num

def append_link_to_txt(name, doi, link, abstract, status, paper_type):
    if not link:
        return
    next_num = count_entries_in_txt() + 1
    lines = [f"\\n{next_num}.{name} - {link}\\n"]
    if doi:
        lines.append(f"DOI: {doi}\\n")
    if status:
        lines.append(f"Status: {status}\\n")
    if paper_type:
        lines.append(f"Type: {paper_type}\\n")
    if abstract:
        lines.append(f" Abstract\\n\\n{abstract}\\n")
    lines.append("\\n")
    with open(LINKS_TXT, \'a\', encoding=\'utf-8\') as f:
        f.writelines(lines)

# ── DB Helper ─────────────────────────────────────────────────────────────────

def add_paper_to_db(name, doi, link, abstract, status, paper_type, df):
    new_row = pd.DataFrame([{
        \'Name\':       name.strip(),
        \'DOI\':        doi.strip() if doi else "",
        \'Link\':       link.strip() if link else "",
        \'Abstract\':   abstract.strip() if abstract else "",
        \'Status\':     status,
        \'Type\':       paper_type,
        \'Date_Added\': datetime.now().strftime("%Y-%m-%d")
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    save_data(df)
    saved_to_txt = False
    if link and not is_link_in_txt(link):
        append_link_to_txt(name, doi, link, abstract, status, paper_type)
        saved_to_txt = True
    return df, saved_to_txt

# ── Helpers: per-field checks ─────────────────────────────────────────────────

def check_name(name, df):
    """Returns (status, message): status in \'ok\',\'duplicate\',\'similar\',\'empty\'"""
    if not name or not name.strip():
        return \'empty\', \'Name is empty.\'
    if name.strip().lower() in [n.lower() for n in df[\'Name\'].values]:
        return \'duplicate\', \'Exact name already exists in the collection.\'
    best_sim, best_match = 0.0, ""
    for n in df[\'Name\'].values:
        s = calculate_similarity(name, str(n))
        if s > best_sim:
            best_sim, best_match = s, str(n)
    if best_sim > 0.8:
        return \'similar\', f\'{best_sim:.0%} similar to: {best_match}\'
    return \'ok\', f\'No duplicate found. (Best match: {best_sim:.0%})\'

def check_doi(doi, df):
    if not doi or not doi.strip():
        return \'empty\', \'DOI is empty.\'
    if doi.strip() in df[\'DOI\'].values:
        return \'duplicate\', \'Exact DOI already exists in the collection.\'
    return \'ok\', \'DOI not found in collection.\'

def check_link(link):
    if not link or not link.strip():
        return \'empty\', \'Link is empty.\'
    if is_link_in_txt(link):
        return \'duplicate\', \'Link already exists in research links new.txt.\'
    return \'ok\', \'Link not found in research links new.txt.\'

def check_abstract(abstract, df):
    if not abstract or not abstract.strip():
        return \'empty\', \'Abstract is empty.\'
    best_sim, best_match = 0.0, ""
    for _, row in df.iterrows():
        if pd.notna(row[\'Abstract\']) and str(row[\'Abstract\']).strip():
            s = calculate_similarity(abstract, str(row[\'Abstract\']))
            if s > best_sim:
                best_sim, best_match = s, str(row[\'Name\'])
    if best_sim > 0.8:
        return \'similar\', f\'{best_sim:.0%} similar to abstract of: {best_match}\'
    return \'ok\', f\'No similar abstract found. (Best match: {best_sim:.0%})\'

def render_check_badge(key):
    """Show the stored check result for a field as a coloured message."""
    result = st.session_state.get(key)
    if result is None:
        return
    status, msg = result
    if status == \'ok\':
        st.sidebar.success(f"OK: {msg}")
    elif status == \'duplicate\':
        st.sidebar.error(f"Duplicate: {msg}")
    elif status == \'similar\':
        st.sidebar.warning(f"Similar: {msg}")
    elif status == \'empty\':
        st.sidebar.info(f"Empty: {msg}")

# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Paper Manager", layout="wide")
    st.title("Research Paper Manager and Duplicate Checker")

    # Session state init
    defaults = {
        \'pending_paper\':    None,
        \'check_all_results\': None,
        \'form_name\':        \'\',
        \'form_doi\':         \'\',
        \'form_link\':        \'\',
        \'form_abstract\':    \'\',
        \'form_type\':        \'Article\',
        \'form_status\':      \'Downloaded\',
        \'chk_name\':         None,
        \'chk_doi\':          None,
        \'chk_link\':         None,
        \'chk_abstract\':     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    def clear_field(field_key, check_key):
        st.session_state[field_key]  = \'\'
        st.session_state[check_key] = None

    def clear_all():
        st.session_state.form_name     = \'\'
        st.session_state.form_doi      = \'\'
        st.session_state.form_link     = \'\'
        st.session_state.form_abstract = \'\'
        st.session_state.form_type     = \'Article\'
        st.session_state.form_status   = \'Downloaded\'
        st.session_state.chk_name      = None
        st.session_state.chk_doi       = None
        st.session_state.chk_link      = None
        st.session_state.chk_abstract  = None

    df = load_data()
    if \'Type\' not in df.columns:
        df[\'Type\'] = \'Article\'
        save_data(df)

    # ── SIDEBAR FORM ──────────────────────────────────────────
    txt_url_count = len(extract_urls_from_txt())
    st.sidebar.markdown(f"📄 **research links new.txt** — `{txt_url_count}` URLs indexed")
    st.sidebar.markdown("---")
    st.sidebar.header("Add New Paper")

    # ── Paper Name ──
    st.sidebar.markdown("**Paper Name \\***")
    c1, c2, c3 = st.sidebar.columns([6, 1, 1])
    with c1:
        name = st.text_input("Paper Name", label_visibility="collapsed",
                             placeholder="e.g. Understanding the Mechanical Biases...",
                             key=\'form_name\')
    with c2:
        if st.button("🔍", key="chk_name_btn", help="Check name against collection"):
            st.session_state.chk_name = check_name(st.session_state.form_name, df)
    with c3:
        if st.button("✕", key="clr_name_btn", help="Clear name"):
            clear_field(\'form_name\', \'chk_name\')
            st.rerun()
    render_check_badge(\'chk_name\')

    # ── DOI ──
    st.sidebar.markdown("**DOI**")
    c1, c2, c3 = st.sidebar.columns([6, 1, 1])
    with c1:
        doi = st.text_input("DOI", label_visibility="collapsed",
                            placeholder="e.g. 10.3390/w13162285",
                            key=\'form_doi\')
    with c2:
        if st.button("🔍", key="chk_doi_btn", help="Check DOI against collection"):
            st.session_state.chk_doi = check_doi(st.session_state.form_doi, df)
    with c3:
        if st.button("✕", key="clr_doi_btn", help="Clear DOI"):
            clear_field(\'form_doi\', \'chk_doi\')
            st.rerun()
    render_check_badge(\'chk_doi\')

    # ── Link ──
    st.sidebar.markdown("**Link (ResearchGate / DOI URL)**")
    c1, c2, c3 = st.sidebar.columns([6, 1, 1])
    with c1:
        link = st.text_input("Link", label_visibility="collapsed",
                             placeholder="https://...",
                             key=\'form_link\')
    with c2:
        if st.button("🔍", key="chk_link_btn", help="Check link against TXT file"):
            st.session_state.chk_link = check_link(st.session_state.form_link)
    with c3:
        if st.button("✕", key="clr_link_btn", help="Clear link"):
            clear_field(\'form_link\', \'chk_link\')
            st.rerun()
    render_check_badge(\'chk_link\')

    # ── Abstract ──
    st.sidebar.markdown("**Abstract (Paste from Notepad)**")
    c1, c2 = st.sidebar.columns([6, 2])
    with c1:
        st.markdown("")   # spacing
    with c2:
        ab_col1, ab_col2 = st.columns(2)
        if ab_col1.button("🔍", key="chk_abs_btn", help="Check abstract similarity", use_container_width=True):
            st.session_state.chk_abstract = check_abstract(st.session_state.form_abstract, df)
        if ab_col2.button("✕", key="clr_abs_btn", help="Clear abstract", use_container_width=True):
            clear_field(\'form_abstract\', \'chk_abstract\')
            st.rerun()
    abstract = st.sidebar.text_area("Abstract", label_visibility="collapsed",
                                    height=180, key=\'form_abstract\')
    render_check_badge(\'chk_abstract\')

    # ── Type & Status ──
    st.sidebar.markdown("**Paper Type:**")
    paper_type = st.sidebar.radio("Choose type", ["Article", "Conference Paper"],
                                  horizontal=True, key=\'form_type\')
    st.sidebar.markdown("**Paper Status:**")
    status = st.sidebar.radio("Choose status",
                              ["Downloaded", "Requested", "Recent (Not on Sci-Hub yet)"],
                              horizontal=True, key=\'form_status\')

    # ── Action Buttons ──
    st.sidebar.markdown("---")
    b1, b2 = st.sidebar.columns(2)
    check_all_fields_clicked = b1.button("Check All Fields", use_container_width=True)
    clear_all_clicked        = b2.button("Clear All Fields", use_container_width=True)

    b3, b4 = st.sidebar.columns(2)
    add_clicked = b3.button("Add Paper", use_container_width=True, type="primary")

    if clear_all_clicked:
        clear_all()
        st.rerun()

    if check_all_fields_clicked:
        st.session_state.chk_name     = check_name(st.session_state.form_name, df)
        st.session_state.chk_doi      = check_doi(st.session_state.form_doi, df)
        st.session_state.chk_link     = check_link(st.session_state.form_link)
        st.session_state.chk_abstract = check_abstract(st.session_state.form_abstract, df)
        st.rerun()

    if add_clicked:
        # Run all checks first
        r_name = check_name(name, df)
        r_doi  = check_doi(doi, df)
        r_link = check_link(link)
        r_abs  = check_abstract(abstract, df)
        st.session_state.chk_name     = r_name
        st.session_state.chk_doi      = r_doi
        st.session_state.chk_link     = r_link
        st.session_state.chk_abstract = r_abs

        # Block on hard errors
        if r_name[0] == \'empty\':
            st.sidebar.error("Paper Name is required!")
        elif r_name[0] == \'duplicate\':
            st.sidebar.error("Duplicate name — paper already in collection.")
        elif r_doi[0] == \'duplicate\':
            st.sidebar.error("Duplicate DOI — paper already in collection.")
        elif r_link[0] == \'duplicate\':
            st.sidebar.error("Link already in research links new.txt.")
        elif r_name[0] == \'similar\' or r_abs[0] == \'similar\':
            # Fuzzy match — queue for confirmation
            best_sim = max(
                (st.session_state.chk_name[0] == \'similar\' and float(st.session_state.chk_name[1].split(\'%\')[0]) / 100) or 0,
                (st.session_state.chk_abstract[0] == \'similar\' and float(st.session_state.chk_abstract[1].split(\'%\')[0]) / 100) or 0
            )
            # find the similar paper row
            sim_row = None
            for _, row in df.iterrows():
                if calculate_similarity(name, row[\'Name\']) > 0.8:
                    sim_row = row
                    break
                if abstract and pd.notna(row[\'Abstract\']) and calculate_similarity(abstract, str(row[\'Abstract\'])) > 0.8:
                    sim_row = row
                    break
            st.session_state.pending_paper = {
                \'name\': name, \'doi\': doi, \'link\': link,
                \'abstract\': abstract, \'status\': status,
                \'paper_type\': paper_type,
                \'sim_row\': sim_row
            }
        else:
            df, saved_to_txt = add_paper_to_db(name, doi, link, abstract, status, paper_type, df)
            st.sidebar.success("Paper added to CSV collection!")
            if saved_to_txt:
                st.sidebar.info("Link also appended to research links new.txt")
            elif link:
                st.sidebar.warning("Link was already in research links new.txt.")
            clear_all()
            st.rerun()

    # ── Pending similarity warning ────────────────────────────
    if st.session_state.pending_paper:
        p = st.session_state.pending_paper
        st.warning("High Similarity Detected — possible duplicate!")
        if p[\'sim_row\'] is not None:
            st.write(f"**Possible duplicate:** {p[\'sim_row\'][\'Name\']}")
        col1, col2 = st.columns(2)
        if col1.button("Confirm and Add Anyway"):
            df, saved_to_txt = add_paper_to_db(
                p[\'name\'], p[\'doi\'], p[\'link\'], p[\'abstract\'], p[\'status\'], p[\'paper_type\'], df
            )
            st.success("Paper added!")
            if saved_to_txt:
                st.info("Link also appended to research links new.txt")
            st.session_state.pending_paper = None
            clear_all()
            st.rerun()
        if col2.button("Cancel"):
            st.session_state.pending_paper = None
            st.rerun()

    # ── Main collection view ──────────────────────────────────
    st.header("Your Paper Collection")
    search = st.text_input("Search by Name, DOI, or Abstract...")
    if search:
        mask = (
            df[\'Name\'].str.contains(search, case=False, na=False) |
            df[\'DOI\'].str.contains(search, case=False, na=False) |
            df[\'Abstract\'].str.contains(search, case=False, na=False)
        )
        view_df = df[mask]
    else:
        view_df = df

    if view_df.empty:
        st.info("Your collection is empty, or no papers match your search. Add one from the sidebar!")
        return

    col_exp, col_chk = st.columns([1, 1])
    with col_exp:
        st.download_button(
            label="Export to CSV",
            data=view_df.to_csv(index=False).encode(\'utf-8\'),
            file_name=\'papers_database_export.csv\',
            mime=\'text/csv\',
            use_container_width=True
        )
    with col_chk:
        if st.button("Check All in TXT", use_container_width=True):
            url_set = extract_urls_from_txt()
            results = []
            for _, row in df.iterrows():
                lnk = str(row.get(\'Link\', \'\')).strip()
                in_txt = is_link_in_txt(lnk, url_set) if lnk else None
                results.append({
                    \'Name\':     row[\'Name\'],
                    \'Link\':     lnk,
                    \'In TXT\':  \'Yes\' if in_txt else (\'No\' if lnk else \'No Link\'),
                    \'DOI\':      str(row.get(\'DOI\', \'\')),
                    \'Status\':   str(row.get(\'Status\', \'\')),
                    \'Type\':     str(row.get(\'Type\', \'\')),
                    \'Abstract\': str(row.get(\'Abstract\', \'\'))
                })
            st.session_state.check_all_results = results

    # Check All Results Panel
    if st.session_state.check_all_results:
        results  = st.session_state.check_all_results
        total    = len(results)
        in_txt   = sum(1 for r in results if r[\'In TXT\'] == \'Yes\')
        missing  = sum(1 for r in results if r[\'In TXT\'] == \'No\')
        no_link  = sum(1 for r in results if r[\'In TXT\'] == \'No Link\')

        st.markdown("---")
        st.subheader("TXT Check Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Papers", total)
        m2.metric("Already in TXT", in_txt)
        m3.metric("Missing from TXT", missing)
        m4.metric("No Link Provided", no_link)

        display_rows = []
        for r in results:
            icon = "Yes" if r[\'In TXT\'] == \'Yes\' else ("No" if r[\'In TXT\'] == \'No\' else "No Link")
            display_rows.append({\'Paper Name\': r[\'Name\'], \'Link\': r[\'Link\'], \'In TXT\': icon})
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        missing_rows = [r for r in results if r[\'In TXT\'] == \'No\']
        if missing_rows:
            if st.button(f"Save All {len(missing_rows)} Missing to TXT", type="primary"):
                url_set = extract_urls_from_txt()
                saved_count = 0
                for r in missing_rows:
                    if r[\'Link\'] and not is_link_in_txt(r[\'Link\'], url_set):
                        append_link_to_txt(r[\'Name\'], r[\'DOI\'], r[\'Link\'], r[\'Abstract\'], r[\'Status\'], r[\'Type\'])
                        url_set.add(r[\'Link\'].strip().rstrip(\'/\').lower())
                        saved_count += 1
                st.success(f"{saved_count} papers appended to research links new.txt!")
                st.session_state.check_all_results = None
                st.rerun()
        else:
            st.success("All papers with links are already in the TXT file!")

        if st.button("Close Results"):
            st.session_state.check_all_results = None
            st.rerun()

    st.markdown("---")
    cols = [\'Name\', \'DOI\', \'Type\', \'Status\', \'Date_Added\']
    st.dataframe(view_df[cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Read Abstracts and Manage")
    url_set_cache = extract_urls_from_txt()

    for index, row in view_df.iterrows():
        paper_type_label = str(row.get(\'Type\', \'\')) if pd.notna(row.get(\'Type\', \'\')) else \'\'
        row_link = str(row.get(\'Link\', \'\')).strip()

        if not row_link:
            txt_badge = "No Link"
        elif is_link_in_txt(row_link, url_set_cache):
            txt_badge = "In TXT"
        else:
            txt_badge = "Not in TXT"

        with st.expander(f"{row[\'Name\']}  |  {paper_type_label}  |  {row[\'Status\']}  |  {txt_badge}"):
            st.markdown(f"**DOI:** `{row[\'DOI\']}`")
            if paper_type_label:
                st.markdown(f"**Type:** {paper_type_label}")
            if row_link:
                st.markdown(f"**Link:** [{row_link}]({row_link})")
            st.markdown(f"**Date Added:** {row[\'Date_Added\']}")
            st.markdown("**Abstract:**")
            st.info(row[\'Abstract\'] if pd.notna(row[\'Abstract\']) else "No abstract provided.")

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if row_link:
                    if is_link_in_txt(row_link, url_set_cache):
                        st.success("Already in TXT")
                    else:
                        if st.button("Save to TXT", key=f"savetxt_{index}"):
                            append_link_to_txt(
                                str(row[\'Name\']), str(row.get(\'DOI\', \'\')),
                                row_link, str(row.get(\'Abstract\', \'\')),
                                str(row.get(\'Status\', \'\')), str(row.get(\'Type\', \'\'))
                            )
                            st.success("Saved to research links new.txt!")
                            st.rerun()
                else:
                    st.info("No link to save")
            with btn_col2:
                if st.button("Delete Paper", key=f"del_{index}"):
                    df = df.drop(index)
                    save_data(df)
                    st.success(f"Deleted \'{row[\'Name\']}\'")
                    st.rerun()

if __name__ == "__main__":
    main()
'''

with open('paper_manager.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("paper_manager.py written successfully.")
