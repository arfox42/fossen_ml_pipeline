# Contains the fields from the MDAT_RANGER CCL message received from a micro modem

std_msgs/Header header

string message_mode

float64 latitude
float64 longitude
float64 depth
float64 speed_kts
uint16  heading

uint8 modem_source
uint8 modem_dest
uint8 mission_leg
uint8 number_legs

string time_remaining
string fix_age
string mission_mode

uint8 battery_pct
bool  major_fault

