from pyparrot.Anafi import Anafi
anafi = Anafi(drone_type="Anafi", ip_address="192.168.42.1")
 
print("Connecting...")
success = anafi.connect(10)
print(success)
print("Sleeping for 5s...")
anafi.smart_sleep(5)
 
print("Take off")
anafi.safe_takeoff(5)
anafi.smart_sleep(1)
 
print("Landing...")
anafi.safe_land(5)
print("DONE - disconnecting")
anafi.disconnect()