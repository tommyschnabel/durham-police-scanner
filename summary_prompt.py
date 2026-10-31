"""
Prompt for the transcript summarizer.

Tuned against Nanbeige 4.2 3B with thinking disabled. Small models copy facts
out of examples placed in the system prompt into unrelated summaries, so the
examples are sent as earlier chat turns and the real transcript is fenced in
<transcript> tags.
"""

# Durham PD / County 10-codes and signals, condensed from the RadioReference
# Durham County (NC) wiki page. DPD's own list (General Order 4014) is not
# public, so a few may be off.
DURHAM_CODES = (
    '10-1 signal weak; 10-2 signal good; 10-3 stop transmitting; 10-4 OK/received; 10-5 '
    'relay; 10-6 busy/stand by; 10-7 out of service (codes: 1 meal, 3 warrant service, 5 '
    'repair, 8 assisting, 9 break, 12 court, 16 at HQ, 17 follow-up, 18 property check, 19 '
    'personal, 20 training); 10-8 in service; 10-9 repeat; 10-10 negative; 10-11 on duty?; '
    '10-12 stand by; 10-13 weather; 10-14 message; 10-15 delivered; 10-16 reply; 10-17 en '
    'route; 10-18 urgent non-emergency; 10-19 in contact; 10-20 location; 10-21 call by '
    'phone; 10-22 disregard; 10-23 arrived on scene; 10-24 assignment complete '
    '(dispositions: 1 investigation report, 3 property report, 4 arrest report, 5 CIT '
    'referral, 6 citation, 7 unfounded, 8 accident report, 10 resolved no report, 12/13 '
    'false alarm, 14 verbal warning, 15 written warning, 16 cleared assisting; suffix F '
    'felony, CS call for service, MI misdemeanor); 10-25 report to location; 10-26 ETA; '
    '10-27 license check; 10-28 registration check; 10-29 stolen/wanted check; 10-30 use '
    'caution; 10-31 pick up; 10-32 backup needed; 10-33 emergency traffic; 10-33a officer '
    'needs assistance; 10-34 time; 10-35 DWI history; 10-36 other charge history; 10-38 '
    'district understaffed; 10-39 comms understaffed; 10-40 fight in progress; 10-41 start '
    'shift; 10-42 end shift; 10-43 car chase; 10-43a foot chase; 10-44 riot; 10-45 bomb '
    'threat; 10-46 bank alarm (green false, yellow no contact, red robbery); 10-47 '
    'expedite; 10-48 detaining subject; 10-49 drag racing; 10-50 traffic crash (PD property'
    ' damage, PI injury, F fatality); 10-51 wrecker; 10-52 ambulance; 10-53 road blocked; '
    '10-54 hit and run; 10-55 drunk driver; 10-56 drunk pedestrian; 10-57 breathalyzer; '
    '10-58 direct traffic; 10-59 escort; 10-60 suspicious vehicle; 10-61 traffic stop; '
    '10-62 breaking and entering; 10-63 investigate; 10-64 crime in progress; 10-65 armed '
    'robbery; 10-66 notify medical examiner; 10-67 death report; 10-68 animal in road; '
    '10-70 improperly parked; 10-72 subject in custody; 10-73 mental health subject; 10-74 '
    'jail break; 10-75 record shows wanted/stolen; 10-76 prowler; 10-77 assist fire with '
    'traffic; 10-78 response (code 1 routine, 2 fast no lights, 3 lights and siren, 5 '
    'cardiac arrest); 10-80 fire; 10-89 burglar alarm; 10-90 officer safety check; 10-91 '
    'rape; 10-92 suicide; 10-93 drugs; 10-94 domestic; 10-97 off radio; 10-98 back on '
    'radio; Signal 20 emergency, hold all traffic; Signal 26 DMV/NCIC down; Signal 29 no '
    'warrant info; Signal 30 is radio secure; Signal 70 bank robbery alarm.\n'
).strip()

SYSTEM_PROMPT = (
    'You summarize police scanner radio traffic from Durham, NC. The transcript comes from '
    'speech-to-text, so expect garbled words and fragments. Provide a brief, factual '
    'summary of key events, incidents, and locations mentioned. Focus on emergency calls, '
    'traffic stops, incidents in progress, officer safety concerns, and notable locations '
    'or descriptions. Be concise: 2-4 sentences, one paragraph, present tense. Do not '
    'speculate beyond the transcript. Output only the summary. Never comment on the '
    'transcript itself, never say what is missing or not mentioned, and never refuse: if '
    'the traffic is garbled, summarize whatever calls, addresses, and units you can make '
    'out. Do not mention Durham or NC in the summary.\n'
    '\n'
    'Reference for decoding radio codes (use it to understand the traffic; do not explain '
    'codes in the summary):\n'
    + DURHAM_CODES + "\n"
    + 'Callsigns like "Charlie 223", "Baker 412", "Medic 31" are units, not people.\n'
    'Each request contains one transcript inside <transcript> tags. Summarize only that '
    'transcript; earlier ones are unrelated.\n'
)

EXAMPLES = [
    (
        '[14:02:10] 10-4, Charlie 214, 10-61 at Roxboro Street and Main, gray Honda Civic, Paul Adam King 4471\n'
        '[14:03:15] Medic 12, 410 Alston Avenue, cross of Pettigrew Street, chest pain, 10-78 code 3\n'
        '[14:03:15] 10-4\n'
        '[14:04:20] Charlie 214, 10-24, code 6',
        'A unit makes a traffic stop on a gray Honda Civic at Roxboro Street and Main and '
        'clears after issuing a citation. Medic 12 responds with lights and siren to a chest '
        'pain call at 410 Alston Avenue near Pettigrew Street.',
    ),
    (
        '[09:10:02] uh 10-4 copy that, Baker 312 going be at the, the store\n'
        '[09:10:30] Medic 7, 1200 Fayetteville Street, fall, uh, cross of Lawson\n'
        '[09:11:14] 312 I got one running, 10-43A, black male red hoodie southbound on\n'
        '[09:11:20] 10-4',
        'Baker 312 starts a foot chase after a black male in a red hoodie heading southbound. '
        'Medic 7 responds to a fall at 1200 Fayetteville Street near Lawson Street.',
    ),
]


def build_messages(transcript_text: str) -> list:
    """Chat messages for one summary request."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for transcript, summary in EXAMPLES:
        messages.append({"role": "user", "content": wrap_transcript(transcript)})
        messages.append({"role": "assistant", "content": summary})
    messages.append({"role": "user", "content": wrap_transcript(transcript_text)})
    return messages


def wrap_transcript(text: str) -> str:
    return f"<transcript>\n{text}\n</transcript>"
